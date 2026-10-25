// SPDX-License-Identifier: GPL-2.0-only
#include "frontend/frame_memory.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <utility>

#ifdef _WIN32
// nss_frontend is also built for the CPU plugin, without the GPU trees' NOMINMAX.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace nss::frontend {
namespace {

// vapoursynth-classic include/VSFrameMemory.h, version 1 (the layout is its
// ABI; the header is not vendored here, since upstream VapourSynth lacks it).
using ReleaseHook = void(VS_CC*)(void* user_data, void* base, std::size_t size);
struct FrameMemoryApi {
    int version;
    std::size_t page_size;
    int(VS_CC* get_plane_block)(const VSFrame* f, int plane, void** base, std::size_t* size) VS_NOEXCEPT;
    int(VS_CC* add_release_hook)(VSCore* core, ReleaseHook hook, void* user_data) VS_NOEXCEPT;
    void(VS_CC* remove_release_hook)(VSCore* core, int id) VS_NOEXCEPT;
};
using GetApi = const FrameMemoryApi*(VS_CC*)(int version);
constexpr int kApiVersion = 1;

const char* env(const FrameMemoryBackend& backend, const char* suffix) {
    return std::getenv((std::string(backend.env) + suffix).c_str());
}

// getVSFrameMemoryAPI from the library that provides the VSAPI in use (the
// Python module may load it with local symbols, so it is not looked up globally).
const FrameMemoryApi* find_api(const FrameMemoryBackend& backend, const VSAPI* vsapi) {
    const char* disabled = env(backend, "");
    if (disabled && std::strcmp(disabled, "0") == 0) return nullptr;
    void* entry = nullptr;
    const void* anchor = reinterpret_cast<const void*>(vsapi->getAPIVersion);
#ifdef _WIN32
    HMODULE module = nullptr;
    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           static_cast<LPCWSTR>(anchor), &module)) {
        entry = reinterpret_cast<void*>(GetProcAddress(module, "getVSFrameMemoryAPI"));
    }
#else
    Dl_info info{};
    if (dladdr(anchor, &info) && info.dli_fname) {
        if (void* library = dlopen(info.dli_fname, RTLD_LAZY | RTLD_NOLOAD)) {
            entry = dlsym(library, "getVSFrameMemoryAPI");
            dlclose(library);  // RTLD_NOLOAD took a reference; the core keeps the library loaded
        }
    }
#endif
    if (!entry) return nullptr;
    const FrameMemoryApi* api = reinterpret_cast<GetApi>(entry)(kApiVersion);
    return api && api->version >= kApiVersion ? api : nullptr;
}

// Seen: transferred once, staged. A block is locked the second time it is
// transferred, so that a pool that churns (a small cache: most blocks are
// used once and released) does not pay a lock and an unlock per frame.
enum class BlockState { Seen, Locking, Locked, Failed };
struct Block {
    std::size_t size;
    VSCore* core;
    BlockState state;
    void* handle;  // the backend's, when Locked
};
// A block once per domain: (base, domain). Integers, so that blocks of one
// base sort together.
using BlockKey = std::pair<std::uintptr_t, std::uintptr_t>;
BlockKey key_of(const void* base, const void* domain) {
    return {reinterpret_cast<std::uintptr_t>(base), reinterpret_cast<std::uintptr_t>(domain)};
}
struct CoreHook {
    int id;
    int instances;
};
// Locked bytes of a core and their cap: half its frame cache, or
// <env>_MAX_MB. At the cap new blocks are staged, so at least half the pool
// stays pageable and a pool that churns locks less.
struct CoreLocks {
    std::size_t cap;
    std::size_t bytes = 0;
};
std::size_t lock_cap(const FrameMemoryBackend& backend, VSCore* core, const VSAPI* vsapi) {
    if (const char* value = env(backend, "_MAX_MB")) {
        return static_cast<std::size_t>(std::strtoull(value, nullptr, 10)) << 20;
    }
    VSCoreInfo info{};
    vsapi->getCoreInfo(core, &info);
    return info.maxFramebufferSize > 0 ? static_cast<std::size_t>(info.maxFramebufferSize) / 2 : 0;
}

// Process-wide: locks are per process (one backend per plugin library). Lock
// order: attach_mu, then VapourSynth's hook lock or mu; the hook runs under
// VapourSynth's hook lock and takes mu, so mu is never held while calling
// into VapourSynth.
struct State {
    std::mutex attach_mu;  // guards cores, domains
    std::map<VSCore*, CoreHook> cores;
    std::map<std::pair<VSCore*, std::uintptr_t>, int> domains;  // instances per (core, domain)
    std::mutex mu;                                              // guards blocks and locks
    std::map<BlockKey, Block> blocks;
    std::map<VSCore*, CoreLocks> locks;
    const FrameMemoryApi* api = nullptr;
    const FrameMemoryBackend* backend = nullptr;
    // <env>_STATS: counts printed when a core's last instance detaches.
    bool stats = false;
    long long locked = 0, failed = 0, hits = 0, misses = 0, first = 0, capped = 0, released = 0;
    double lock_ms = 0, unlock_ms = 0, unlock_max_ms = 0;
    std::size_t locked_bytes = 0, locked_peak = 0;
};
State& state() {
    static State* s = new State;  // never destroyed: hooks may run during process teardown
    return *s;
}

double since_ms(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// Under s.mu: unlocks a Locked block (the entry stays).
void unlock_block(State& s, const BlockKey& key, Block& block) {
    if (block.state != BlockState::Locked) return;
    s.backend->unlock(reinterpret_cast<const void*>(key.second), reinterpret_cast<void*>(key.first), block.size,
                      block.handle);
    block.state = BlockState::Seen;
    block.handle = nullptr;
    s.locked_bytes -= block.size;
    s.locks[block.core].bytes -= block.size;
}

void VS_CC on_release(void*, void* base, std::size_t) {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    // Every domain's entry of the block. Never Locking: the locking thread
    // holds a frame whose plane is in the block.
    auto it = s.blocks.lower_bound(key_of(base, nullptr));
    while (it != s.blocks.end() && it->first.first == reinterpret_cast<std::uintptr_t>(base)) {
        if (it->second.state == BlockState::Locked) {
            const auto t0 = std::chrono::steady_clock::now();
            unlock_block(s, it->first, it->second);
            const double ms = since_ms(t0);
            ++s.released;
            s.unlock_ms += ms;
            s.unlock_max_ms = std::max(s.unlock_max_ms, ms);
        }
        it = s.blocks.erase(it);
    }
}

}  // namespace

FrameMemory::FrameMemory(const FrameMemoryBackend& backend, VSCore* core, const VSAPI* vsapi, const void* domain)
    : backend_(&backend), domain_(domain) {
    State& s = state();
    static std::once_flag once;
    std::call_once(once, [&] {
        s.backend = &backend;
        s.stats = env(backend, "_STATS") != nullptr;
        s.api = find_api(backend, vsapi);
    });
    if (!s.api || s.backend != &backend) return;
    std::lock_guard<std::mutex> lock(s.attach_mu);
    auto it = s.cores.find(core);
    if (it == s.cores.end()) {
        const int id = s.api->add_release_hook(core, on_release, nullptr);
        if (id <= 0) return;
        it = s.cores.emplace(core, CoreHook{id, 0}).first;
        const std::size_t cap = lock_cap(backend, core, vsapi);  // calls VapourSynth: not under s.mu
        std::lock_guard<std::mutex> locks(s.mu);
        s.locks[core] = CoreLocks{cap};
    }
    ++it->second.instances;
    ++s.domains[{core, reinterpret_cast<std::uintptr_t>(domain)}];
    core_ = core;
}

FrameMemory::~FrameMemory() {
    if (!core_) return;
    State& s = state();
    std::lock_guard<std::mutex> attach(s.attach_mu);
    const std::pair<VSCore*, std::uintptr_t> domain{core_, reinterpret_cast<std::uintptr_t>(domain_)};
    if (--s.domains[domain] == 0) {
        s.domains.erase(domain);
        // Nothing of this domain transfers frames of this core any more:
        // unlock its blocks while the hook still covers them (the domain, a
        // device, may go away next).
        std::lock_guard<std::mutex> lock(s.mu);
        for (auto b = s.blocks.begin(); b != s.blocks.end();) {
            if (b->second.core != core_ || b->first.second != domain.second) {
                ++b;
                continue;
            }
            unlock_block(s, b->first, b->second);
            b = s.blocks.erase(b);
        }
    }
    auto it = s.cores.find(core_);
    if (--it->second.instances > 0) return;
    const int id = it->second.id;
    s.cores.erase(it);
    {
        std::lock_guard<std::mutex> lock(s.mu);
        s.locks.erase(core_);
        if (s.stats) {
            std::fprintf(stderr,
                         "%s direct I/O: %lld blocks locked (%.1f ms), %lld failed, %lld direct transfers, "
                         "%lld staged at first use, %lld staged at the lock cap, %lld staged (block busy or failed), "
                         "%lld released by the pool (unlock %.1f ms in all, longest %.2f ms, on the freeing thread); "
                         "peak locked %.0f MiB\n",
                         backend_->name, s.locked, s.lock_ms, s.failed, s.hits, s.first, s.capped, s.misses,
                         s.released, s.unlock_ms, s.unlock_max_ms, s.locked_peak / 1048576.0);
        }
    }
    // Not under s.mu: removal waits for a running hook, which takes s.mu.
    s.api->remove_release_hook(core_, id);
}

PinnedPlane FrameMemory::pinned(const VSFrame* f, int plane, const void* data) {
    if (!core_) return {};
    State& s = state();
    void* base = nullptr;
    std::size_t size = 0;
    if (!s.api->get_plane_block(f, plane, &base, &size)) return {};
    std::size_t offset = 0;
    if (data) {
        const auto at = reinterpret_cast<std::uintptr_t>(data), start = reinterpret_cast<std::uintptr_t>(base);
        if (at < start || at - start >= size) return {};
        offset = at - start;
    }
    const BlockKey key = key_of(base, domain_);
    {
        std::lock_guard<std::mutex> lock(s.mu);
        const auto [it, inserted] = s.blocks.try_emplace(key, Block{size, core_, BlockState::Seen, nullptr});
        if (inserted) {
            ++s.first;
            return {};
        }
        Block& block = it->second;
        if (block.state != BlockState::Seen || block.size != size) {
            const bool hit = block.state == BlockState::Locked && block.size == size;
            ++(hit ? s.hits : s.misses);
            return hit ? PinnedPlane{block.handle, offset} : PinnedPlane{};
        }
        CoreLocks& locks = s.locks[core_];
        if (locks.bytes + size > locks.cap) {
            ++s.capped;
            return {};  // stays Seen: locked once the core is below its cap again
        }
        locks.bytes += size;  // reserved now, so that concurrent locks stay within the cap
        block.state = BlockState::Locking;
    }
    // Outside the lock: locking a 1080p float plane takes about 0.6 ms (CUDA).
    const auto t0 = std::chrono::steady_clock::now();
    void* handle = backend_->lock(domain_, base, size);
    const double ms = since_ms(t0);
    std::lock_guard<std::mutex> lock(s.mu);
    Block& block = s.blocks[key];
    block.state = handle ? BlockState::Locked : BlockState::Failed;
    block.handle = handle;
    if (handle) {
        s.locked_bytes += size;
        s.locked_peak = std::max(s.locked_peak, s.locked_bytes);
    } else {
        s.locks[core_].bytes -= size;
    }
    ++(handle ? s.locked : s.failed);
    ++(handle ? s.hits : s.misses);
    s.lock_ms += ms;
    return handle ? PinnedPlane{handle, offset} : PinnedPlane{};
}

}  // namespace nss::frontend
