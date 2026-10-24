// SPDX-License-Identifier: GPL-2.0-only
// Direct DMA between the device and VapourSynth frames, where the host
// offers vapoursynth-classic's VSFrameMemory.h extension. Frame planes live in
// blocks of the core's frame pool, which are reused for later frames: a block
// is page-locked with cudaHostRegister the second time a plane in it is
// transferred (the first is staged: a pool that churns would otherwise pay a
// lock and an unlock per frame), and unlocked by the pool's release hook
// before the block goes back to the system. Transfers from and to such planes
// then skip the pinned staging copies (frame_io.hpp upload_direct /
// download_direct). At most half of a core's frame cache is locked
// (NSS_CUDA_DIRECT_IO_MAX_MB overrides): beyond it, blocks are staged.
//
// Without the extension (upstream VapourSynth, or NSS_CUDA_DIRECT_IO=0 in the
// environment) pinned() is always false and the filters stage as before. The
// bytes moved are the same either way, so outputs are identical.
#pragma once

#include <VapourSynth4.h>

namespace nss_cuda {

// One per filter instance, created on the instance's device.
class FrameMemory {
public:
    // Attaches to core's frame pool (the first instance on a core adds the
    // release hook).
    FrameMemory(VSCore* core, const VSAPI* vsapi);
    // The last instance on a core unlocks every block it locked and removes
    // the hook: the pool may outlive the core and this plugin.
    ~FrameMemory();
    FrameMemory(const FrameMemory&) = delete;
    FrameMemory& operator=(const FrameMemory&) = delete;

    // True when plane `plane` of f is page-locked, so that a DMA may read or
    // write it directly while f is held. Locks the plane's block on its second
    // use (about 0.6 ms for a 1080p float plane); false at the first use, while
    // another thread is locking the same block, when locking failed, and
    // without the extension.
    bool pinned(const VSFrame* f, int plane);

private:
    VSCore* core_ = nullptr;  // nullptr: not available
};

}  // namespace nss_cuda
