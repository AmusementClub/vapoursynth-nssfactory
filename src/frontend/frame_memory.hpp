// SPDX-License-Identifier: GPL-2.0-only
// Direct DMA between a device and VapourSynth frames, where the host offers
// vapoursynth-classic's VSFrameMemory.h extension: the backend-neutral part,
// shared by nss_cuda (cudaHostRegister) and nss_vk (VK_EXT_external_memory_host
// imports). Frame planes live in blocks of the core's frame pool, which are
// reused for later frames: a block is locked for DMA by the backend the second
// time a plane in it is transferred (the first is staged: a pool that churns
// would otherwise pay a lock and an unlock per frame), and unlocked by the
// pool's release hook before the block goes back to the system. At most half
// of a core's frame cache is locked (<env>_MAX_MB overrides): beyond it,
// blocks are staged.
//
// Without the extension (upstream VapourSynth, or <env>=0 in the environment)
// pinned() never locks anything and the filters stage as before. The bytes
// moved are the same either way, so outputs are identical.
#pragma once

#include <VapourSynth4.h>

#include <cstddef>

namespace nss::frontend {

// What a backend does to a pool block. `domain` names what a lock serves: a
// device, or nullptr when one lock serves every device of the process.
struct FrameMemoryBackend {
    // Environment prefix: <env>=0 disables direct I/O, <env>_MAX_MB caps the
    // locked bytes per core, <env>_STATS prints counts when a core's last
    // instance detaches.
    const char* env;
    const char* name;  // starts the stats line
    // Makes [base, base + size) usable for DMA by `domain`: a non-null handle,
    // or nullptr when that failed (the block is then staged for good).
    void* (*lock)(const void* domain, void* base, std::size_t size);
    // Undoes lock(), once per handle: on the thread that frees the block
    // (under VapourSynth's hook lock: call nothing of VapourSynth's), or in
    // the destructor of the domain's last FrameMemory on the core.
    void (*unlock)(const void* domain, void* base, std::size_t size, void* handle);
};

// The lock of the block holding a plane, and the plane's place in it.
struct PinnedPlane {
    void* handle = nullptr;  // the backend's; null: stage this plane
    std::size_t offset = 0;  // of the plane's first byte in the block
    explicit operator bool() const noexcept { return handle != nullptr; }
};

// One per filter instance. `backend` must outlive every instance (a static).
class FrameMemory {
public:
    // Attaches to core's frame pool (the first instance on a core adds the
    // release hook).
    FrameMemory(const FrameMemoryBackend& backend, VSCore* core, const VSAPI* vsapi, const void* domain = nullptr);
    // The domain's last instance on a core unlocks every block locked for it,
    // the core's last also removes the hook: the pool may outlive the core
    // and this plugin.
    ~FrameMemory();
    FrameMemory(const FrameMemory&) = delete;
    FrameMemory& operator=(const FrameMemory&) = delete;

    // The lock of the block holding plane `plane` of f, so that a DMA may read
    // or write the plane while f is held. Locks the block on its second use;
    // empty at the first use, while another thread is locking the same block,
    // when locking failed, at the cap, and without the extension. `data` is
    // the plane pointer the caller transfers (getReadPtr, or getWritePtr for
    // a frame it writes: call that first, it may move a shared plane); the
    // offset is taken from it, and a pointer outside the block is staged.
    // Without `data` the offset is 0.
    PinnedPlane pinned(const VSFrame* f, int plane, const void* data = nullptr);
    // The extension is there (pinned() may lock).
    bool available() const noexcept { return core_ != nullptr; }

private:
    const FrameMemoryBackend* backend_;
    VSCore* core_ = nullptr;  // nullptr: not available
    const void* domain_;
};

}  // namespace nss::frontend
