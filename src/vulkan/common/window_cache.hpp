// SPDX-License-Identifier: GPL-2.0-only
// Frames of a clip kept on the device for the requests whose temporal
// windows read them, shared by a filter's slots (NLM's tile path, PNLM): a
// frame is uploaded once however many windows it is part of. A request pins
// its whole window at once (never waiting while it holds pins), takes the
// cache's free frames, those used longest ago, for the frames that are not
// there, and uploads them; requests that read a frame another one uploads wait
// for that upload on the device. Frames live in regions of one PlaneStore.
#pragma once

#include "frontend/temporal.hpp"
#include "vulkan/runtime/memory.hpp"
#include "vulkan/runtime/vk.hpp"

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nss_vk {
// A frame of the clip in the cache (region `region` of the store).
// Guarded by WindowCache::mu.
struct WindowFrame {
    int region = 0;
    int n = -1;           // frame number, -1: none
    int pins = 0;         // requests reading it; it is not replaced while pinned
    bool queued = false;  // written, and its upload (if any) submitted
    bool failed = false;  // the request that was to upload it gave up
    std::uint64_t used = 0;
    // The upload stage that copies it to the device (null with BAR memory).
    VkSemaphore ready = VK_NULL_HANDLE;
    std::uint64_t ready_value = 0;
    // The stages that last used the region, one per slot timeline (the upload
    // that wrote it, the compute stages that read it): an upload stage that
    // copies another frame into the region waits for them on the device. The
    // host has waited for them already (a region is reused only once no
    // request pins it), but only a device-side dependency orders the copies
    // of different slots into one region (synchronization validation:
    // write-after-write on the store, with staging or imported frames).
    std::vector<std::pair<VkSemaphore, std::uint64_t>> uses;
};

struct WindowCache {
    std::string filter = "nss_vk";  // error prefix: "nss_vk.<filter>"
    std::mutex mu;
    std::condition_variable changed;
    std::vector<std::unique_ptr<WindowFrame>> frames;
    std::uint64_t clock = 0;
    PlaneStore store;  // the frames' planes
};

// The window of one request: pins its frames in the cache
// and takes the cache's free frames for those that are not there. Unpins on
// destruction; a frame it did not get to upload is given up (as nss_cuda).
class Window {
public:
    Window(WindowCache& cache, int n, int radius, int clip_frames) : cache_(cache), slots_(2 * radius + 1) {
        std::vector<int> wanted;
        for (int t = 0; t < 2 * radius + 1; ++t) {
            const int fn = nss::host_detail::temporal_slot_frame(n, t, radius, clip_frames);
            if (std::find(wanted.begin(), wanted.end(), fn) == wanted.end()) wanted.push_back(fn);
        }
        std::unique_lock lock(cache_.mu);
        std::vector<WindowFrame*> found(wanted.size());
        // All of the window at once: a request never waits while it holds pins.
        cache_.changed.wait(lock, [&] {
            std::size_t missing = 0, free = 0;
            for (std::size_t i = 0; i < wanted.size(); ++i) {
                found[i] = nullptr;
                for (auto& f : cache_.frames) {
                    if (f->n == wanted[i]) found[i] = f.get();
                }
                missing += found[i] == nullptr;
            }
            for (auto& f : cache_.frames) {
                free += f->pins == 0 && std::find(found.begin(), found.end(), f.get()) == found.end();
            }
            return missing <= free;
        });
        for (std::size_t i = 0; i < wanted.size(); ++i) {
            WindowFrame* f = found[i];
            if (!f) {
                // The free frame that was used longest ago.
                for (auto& other : cache_.frames) {
                    if (other->pins != 0 || std::find(found.begin(), found.end(), other.get()) != found.end()) continue;
                    if (!f || other->used < f->used) f = other.get();
                }
                f->n = wanted[i];
                f->queued = false;
                f->failed = false;
                f->ready = VK_NULL_HANDLE;
                f->ready_value = 0;
                found[i] = f;
                uploads_.push_back(f);
            }
            ++f->pins;
            f->used = ++cache_.clock;
            pinned_.push_back(f);
        }
        for (int t = 0; t < 2 * radius + 1; ++t) {
            const int fn = nss::host_detail::temporal_slot_frame(n, t, radius, clip_frames);
            slots_[t] = found[std::find(wanted.begin(), wanted.end(), fn) - wanted.begin()];
        }
    }
    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;
    ~Window() {
        {
            std::lock_guard lock(cache_.mu);
            for (WindowFrame* f : uploads_) {
                if (f->queued) continue;
                f->n = -1;
                f->failed = true;
            }
            for (WindowFrame* f : pinned_) --f->pins;
        }
        cache_.changed.notify_all();
    }

    WindowFrame& slot(int t) const { return *slots_[t]; }
    const std::vector<WindowFrame*>& uploads() const { return uploads_; }
    bool uploads(const WindowFrame* f) const { return std::find(uploads_.begin(), uploads_.end(), f) != uploads_.end(); }
    const std::vector<WindowFrame*>& pinned() const { return pinned_; }

    // Returns once the frames other requests upload are queued; their upload
    // stages are then known for the device-side waits.
    void wait_queued() {
        std::unique_lock lock(cache_.mu);
        bool failed = false;
        cache_.changed.wait(lock, [&] {
            bool all = true;
            for (WindowFrame* f : pinned_) {
                failed = failed || f->failed;
                all = all && (f->queued || uploads(f));
            }
            return failed || all;
        });
        if (failed) throw std::runtime_error(cache_.filter + ": a frame of the window was not uploaded");
    }
    void mark_queued(VkSemaphore ready, std::uint64_t value) {
        {
            std::lock_guard lock(cache_.mu);
            for (WindowFrame* f : uploads_) {
                f->queued = true;
                f->ready = ready;
                f->ready_value = value;
                // The upload waited for the region's earlier uses (or the
                // host wrote it after they were done).
                f->uses.clear();
                if (ready) f->uses.emplace_back(ready, value);
            }
        }
        cache_.changed.notify_all();
    }
    // Before the uploads are recorded: the stages that last used their regions.
    std::vector<std::pair<VkSemaphore, std::uint64_t>> previous_uses() const {
        std::lock_guard lock(cache_.mu);
        std::vector<std::pair<VkSemaphore, std::uint64_t>> uses;
        for (const WindowFrame* f : uploads_) uses.insert(uses.end(), f->uses.begin(), f->uses.end());
        return uses;
    }
    // After the submission: its compute stage read every frame of the window.
    void read_by(std::pair<VkSemaphore, std::uint64_t> compute) {
        if (!compute.first) return;
        std::lock_guard lock(cache_.mu);
        for (WindowFrame* f : pinned_) {
            auto use = std::find_if(f->uses.begin(), f->uses.end(), [&](const auto& u) { return u.first == compute.first; });
            if (use == f->uses.end()) {
                f->uses.push_back(compute);
            } else {
                use->second = std::max(use->second, compute.second);
            }
        }
    }

private:
    WindowCache& cache_;
    std::vector<WindowFrame*> slots_, pinned_, uploads_;
};

}  // namespace nss_vk
