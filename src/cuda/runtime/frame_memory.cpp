// SPDX-License-Identifier: GPL-2.0-only
#include "cuda/runtime/frame_memory.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace nss_cuda {
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

// getVSFrameMemoryAPI from the library that provides the VSAPI in use (the
// Python module may load it with local symbols, so it is not looked up globally).
const FrameMemoryApi* find_api(const VSAPI* vsapi) {
    const char* env = std::getenv("NSS_CUDA_DIRECT_IO");
    if (env && std::strcmp(env, "0") == 0) return nullptr;
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
};
struct CoreHook {
    int id;
    int instances;
};
// Page-locked bytes of a core and their cap: half its frame cache, or
// NSS_CUDA_DIRECT_IO_MAX_MB. At the cap new blocks are staged, so at least
// half the pool stays pageable and a pool that churns locks less.
struct CoreLocks {
    std::size_t cap;
    std::size_t bytes = 0;
};
std::size_t lock_cap(VSCore* core, const VSAPI* vsapi) {
    if (const char* env = std::getenv("NSS_CUDA_DIRECT_IO_MAX_MB")) {
        return static_cast<std::size_t>(std::strtoull(env, nullptr, 10)) << 20;
    }
    VSCoreInfo info{};
    vsapi->getCoreInfo(core, &info);
    return info.maxFramebufferSize > 0 ? static_cast<std::size_t>(info.maxFramebufferSize) / 2 : 0;
}

// Process-wide: page locks are per process. Lock order: attach_mu, then
// VapourSynth's hook lock or mu; the hook runs under VapourSynth's hook lock
// and takes mu, so mu is never held while calling into VapourSynth.
struct State {
    std::mutex attach_mu;               // guards cores
    std::map<VSCore*, CoreHook> cores;
    std::mutex mu;                      // guards blocks and locks
    std::map<void*, Block> blocks;
    std::map<VSCore*, CoreLocks> locks;
    const FrameMemoryApi* api = nullptr;
    // NSS_CUDA_DIRECT_IO_STATS: counts printed when a core's last instance detaches.
    bool stats = std::getenv("NSS_CUDA_DIRECT_IO_STATS") != nullptr;
    long long locked = 0, failed = 0, hits = 0, misses = 0, first = 0, capped = 0, released = 0;
    double lock_ms = 0, unlock_ms = 0, unlock_max_ms = 0;
    std::size_t locked_bytes = 0, locked_peak = 0;
};
State& state() {
    static State* s = new State;  // never destroyed: hooks may run during process teardown
    return *s;
}

void VS_CC on_release(void*, void* base, std::size_t) {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    const auto it = s.blocks.find(base);
    if (it == s.blocks.end()) return;
    // Never Locking: the locking thread holds a frame whose plane is in the block.
    if (it->second.state == BlockState::Locked) {
        const auto t0 = std::chrono::steady_clock::now();
        cudaHostUnregister(base);
        cudaGetLastError();
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        ++s.released;
        s.locked_bytes -= it->second.size;
        s.locks[it->second.core].bytes -= it->second.size;
        s.unlock_ms += ms;
        s.unlock_max_ms = std::max(s.unlock_max_ms, ms);
    }
    s.blocks.erase(it);
}

}  // namespace

FrameMemory::FrameMemory(VSCore* core, const VSAPI* vsapi) {
    static std::once_flag once;
    std::call_once(once, [&] { state().api = find_api(vsapi); });
    State& s = state();
    if (!s.api) return;
    std::lock_guard<std::mutex> lock(s.attach_mu);
    auto it = s.cores.find(core);
    if (it == s.cores.end()) {
        const int id = s.api->add_release_hook(core, on_release, nullptr);
        if (id <= 0) return;
        it = s.cores.emplace(core, CoreHook{id, 0}).first;
        const std::size_t cap = lock_cap(core, vsapi);  // calls VapourSynth: not under s.mu
        {
            std::lock_guard<std::mutex> lock(s.mu);
            s.locks[core] = CoreLocks{cap};
        }
    }
    ++it->second.instances;
    core_ = core;
}

FrameMemory::~FrameMemory() {
    if (!core_) return;
    State& s = state();
    std::lock_guard<std::mutex> attach(s.attach_mu);
    auto it = s.cores.find(core_);
    if (--it->second.instances > 0) return;
    const int id = it->second.id;
    s.cores.erase(it);
    {
        // No instance is left to transfer frames of this core: unlock its
        // blocks while the hook still covers them, then remove the hook.
        std::lock_guard<std::mutex> lock(s.mu);
        for (auto b = s.blocks.begin(); b != s.blocks.end();) {
            if (b->second.core != core_) {
                ++b;
                continue;
            }
            if (b->second.state == BlockState::Locked) {
                cudaHostUnregister(b->first);
                cudaGetLastError();
                s.locked_bytes -= b->second.size;
            }
            b = s.blocks.erase(b);
        }
        s.locks.erase(core_);
        if (s.stats) {
            std::fprintf(stderr,
                         "nss_cuda direct I/O: %lld blocks locked (%.1f ms), %lld failed, %lld direct transfers, "
                         "%lld staged at first use, %lld staged at the lock cap, %lld staged (block busy or failed), "
                         "%lld released by the pool (unlock %.1f ms in all, longest %.2f ms, on the freeing thread); "
                         "peak locked %.0f MiB\n",
                         s.locked, s.lock_ms, s.failed, s.hits, s.first, s.capped, s.misses, s.released, s.unlock_ms,
                         s.unlock_max_ms, s.locked_peak / 1048576.0);
        }
    }
    // Not under s.mu: removal waits for a running hook, which takes s.mu.
    s.api->remove_release_hook(core_, id);
}

bool FrameMemory::pinned(const VSFrame* f, int plane) {
    if (!core_) return false;
    State& s = state();
    void* base = nullptr;
    std::size_t size = 0;
    if (!s.api->get_plane_block(f, plane, &base, &size)) return false;
    {
        std::lock_guard<std::mutex> lock(s.mu);
        const auto [it, inserted] = s.blocks.try_emplace(base, Block{size, core_, BlockState::Seen});
        if (inserted) {
            ++s.first;
            return false;
        }
        if (it->second.state != BlockState::Seen || it->second.size != size) {
            const bool hit = it->second.state == BlockState::Locked && it->second.size == size;
            ++(hit ? s.hits : s.misses);
            return hit;
        }
        CoreLocks& locks = s.locks[core_];
        if (locks.bytes + size > locks.cap) {
            ++s.capped;
            return false;  // stays Seen: locked once the core is below its cap again
        }
        locks.bytes += size;  // reserved now, so that concurrent locks stay within the cap
        it->second.state = BlockState::Locking;
    }
    // Outside the lock: page-locking a 1080p plane takes about 0.6 ms.
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = cudaHostRegister(base, size, cudaHostRegisterPortable) == cudaSuccess;
    if (!ok) cudaGetLastError();
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::lock_guard<std::mutex> lock(s.mu);
    s.blocks[base].state = ok ? BlockState::Locked : BlockState::Failed;
    if (!ok) s.locks[core_].bytes -= size;
    if (ok) {
        s.locked_bytes += size;
        s.locked_peak = std::max(s.locked_peak, s.locked_bytes);
    }
    ++(ok ? s.locked : s.failed);
    ++(ok ? s.hits : s.misses);
    s.lock_ms += ms;
    return ok;
}

}  // namespace nss_cuda
