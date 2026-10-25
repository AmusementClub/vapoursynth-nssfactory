// SPDX-License-Identifier: GPL-2.0-only
// Direct DMA between the device and VapourSynth frames, where the host
// offers vapoursynth-classic's VSFrameMemory.h extension: the CUDA backend of
// frontend/frame_memory.hpp, which tracks the core's pool blocks (locked on
// their second transfer, unlocked by the pool's release hook, at most half of
// the frame cache; NSS_CUDA_DIRECT_IO_MAX_MB overrides). A block is
// page-locked with cudaHostRegister (portable: one lock serves every device).
// Transfers from and to such planes then skip the pinned staging copies
// (frame_io.hpp upload_direct / download_direct).
//
// Without the extension (upstream VapourSynth, or NSS_CUDA_DIRECT_IO=0 in the
// environment) pinned() is always false and the filters stage as before. The
// bytes moved are the same either way, so outputs are identical.
// NSS_CUDA_DIRECT_IO_STATS prints counts when a core's last instance goes.
#pragma once

#include "frontend/frame_memory.hpp"

#include <VapourSynth4.h>

namespace nss_cuda {

// One per filter instance, created on the instance's device.
class FrameMemory {
public:
    FrameMemory(VSCore* core, const VSAPI* vsapi);

    // True when plane `plane` of f is page-locked, so that a DMA may read or
    // write it directly while f is held. Locks the plane's block on its second
    // use (about 0.6 ms for a 1080p float plane); false at the first use, while
    // another thread is locking the same block, when locking failed, and
    // without the extension.
    bool pinned(const VSFrame* f, int plane) { return static_cast<bool>(memory_.pinned(f, plane)); }
    // The extension is there (pinned() may return true).
    bool available() const { return memory_.available(); }

private:
    nss::frontend::FrameMemory memory_;
};

}  // namespace nss_cuda
