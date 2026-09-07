#pragma once

#include <VapourSynth4.h>
#include "nss/checked.hpp"
#include "nss/resources.hpp"
#include <algorithm>
#include <array>
#include <cstring>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

namespace nss {

class NodeRef {
public:
    NodeRef(std::nullptr_t = nullptr) noexcept {}
    NodeRef(VSNode* node, const VSAPI* api) noexcept : node_(node), api_(api) {}
    NodeRef(const NodeRef&) = delete;
    NodeRef& operator=(const NodeRef&) = delete;
    NodeRef(NodeRef&& other) noexcept : node_(other.release()), api_(other.api_) {}
    NodeRef& operator=(NodeRef&& other) noexcept {
        if (this != &other) { reset(); api_ = other.api_; node_ = other.release(); }
        return *this;
    }
    NodeRef& operator=(std::nullptr_t) noexcept { reset(); return *this; }
    ~NodeRef() { reset(); }
    operator VSNode*() const noexcept { return node_; }
    VSNode* get() const noexcept { return node_; }
    VSNode* release() noexcept { return std::exchange(node_, nullptr); }
    void reset() noexcept { if (node_) api_->freeNode(release()); }
private:
    VSNode* node_ = nullptr;
    const VSAPI* api_ = nullptr;
};

inline NodeRef get_node(const VSAPI* api, const VSMap* map, const char* key, int index, int* error) {
    return {api->mapGetNode(map, key, index, error), api};
}

// Tracks references, including multiple references to the same frame. Most
// callbacks fit in inline storage; larger temporal windows spill safely.
class FrameScope {
public:
    explicit FrameScope(const VSAPI* api) : api_(api),
        outputs_(make_resource_account(ResourceKind::Output)),
        framework_(make_resource_account(ResourceKind::Framework)) {}
    FrameScope(const FrameScope&) = delete;
    FrameScope& operator=(const FrameScope&) = delete;
    ~FrameScope() {
        for (auto& entry : inline_) if (entry.frame) { uncharge(entry); api_->freeFrame(entry.frame); }
        for (auto& entry : spill_) if (entry.frame) { uncharge(entry); api_->freeFrame(entry.frame); }
    }
    const VSFrame* getFrameFilter(int n, VSNode* node, VSFrameContext* context) {
        return hold(api_->getFrameFilter(n, node, context));
    }
    VSFrame* newVideoFrame(const VSVideoFormat* format, int w, int h, const VSFrame* props, VSCore* core) {
        checked_int(checked_mul(w, h));
        return hold(api_->newVideoFrame(format, w, h, props, core), true);
    }
    VSFrame* newVideoFrame2(const VSVideoFormat* format, int w, int h, const VSFrame** frames,
                            const int* planes, const VSFrame* props, VSCore* core) {
        checked_int(checked_mul(w, h));
        return hold(api_->newVideoFrame2(format, w, h, frames, planes, props, core), true);
    }
    // Existing group APIs use one stride per channel across time. Retain the
    // direct view when it fits that contract, otherwise adapt only that plane.
    const float* readPlane(const VSFrame* frame, int plane, int expected_stride) {
        const int actual = static_cast<int>(getStride(frame, plane) / sizeof(float));
        const auto* source = reinterpret_cast<const float*>(api_->getReadPtr(frame, plane));
        if (actual == expected_stride) return source;
        const int height = api_->getFrameHeight(frame, plane);
        const int width = api_->getFrameWidth(frame, plane);
        if (expected_stride < width) throw std::invalid_argument("nss: incompatible plane stride");
        for (const auto& mirror : mirrors_)
            if (mirror.frame == frame && mirror.plane == plane && mirror.stride == expected_stride)
                return mirror.pixels.data();
        Mirror mirror{frame, plane, expected_stride, {}};
        mirror.pixels.resize(checked_mul(expected_stride, height));
        for (int row = 0; row < height; ++row)
            std::memcpy(mirror.pixels.data() + static_cast<std::size_t>(row) * expected_stride,
                        source + static_cast<std::size_t>(row) * actual, width * sizeof(float));
        mirrors_.push_back(std::move(mirror));
        return mirrors_.back().pixels.data();
    }
    std::ptrdiff_t getStride(const VSFrame* frame, int plane) const {
        const auto bytes = api_->getStride(frame, plane);
        const int height = api_->getFrameHeight(frame, plane);
        const int width = api_->getFrameWidth(frame, plane);
        if (bytes <= 0 || bytes % sizeof(float) || height < 1 || width < 1 ||
            static_cast<std::size_t>(bytes) / sizeof(float) < static_cast<std::size_t>(width))
            throw std::invalid_argument("nss: invalid float frame stride");
        checked_int(checked_mul(static_cast<std::size_t>(bytes) / sizeof(float), height));
        return bytes;
    }
    void freeFrame(const VSFrame* frame) noexcept { if (frame) { forget(frame); api_->freeFrame(frame); } }
    template<class T> T* keep(T* frame) {
        if (current_budget()) {
            const auto stats = current_budget()->snapshot();
            VSMap* props = api_->getFramePropertiesRW(frame);
            std::int64_t bytes[resource_kinds];
            for (std::size_t i = 0; i < resource_kinds; ++i) bytes[i] = static_cast<std::int64_t>(stats.bytes[i]);
            if (api_->mapSetIntArray(props, "_NSSResourceBytes", bytes, resource_kinds) ||
                api_->mapSetInt(props, "_NSSResourcePeak", static_cast<std::int64_t>(stats.peak), maReplace) ||
                api_->mapSetInt(props, "_NSSResourceLimit", static_cast<std::int64_t>(stats.limit), maReplace))
                throw std::bad_alloc();
        }
        forget(frame); return frame;
    }
private:
    struct Entry { const VSFrame* frame = nullptr; std::size_t bytes = 0; bool output = false; };
    void uncharge(const Entry& entry) noexcept {
        const auto& account = entry.output ? outputs_ : framework_;
        if (account) account->release(entry.bytes);
    }
    template<class T> T* hold(T* frame, bool output = false) {
        if (!frame) throw std::bad_alloc();
        Entry entry{frame, 0, output};
        const auto& account = output ? outputs_ : framework_;
        bool charged = false;
        try {
            if (account) {
                const auto* format = api_->getVideoFrameFormat(frame);
                for (int plane = 0; plane < format->numPlanes; ++plane)
                    entry.bytes = checked_add(entry.bytes, checked_mul(api_->getStride(frame, plane),
                                                                        api_->getFrameHeight(frame, plane)));
                account->acquire(entry.bytes); charged = true;
            }
            for (auto& slot : inline_) if (!slot.frame) { slot = entry; return frame; }
            for (auto& slot : spill_) if (!slot.frame) { slot = entry; return frame; }
            spill_.push_back(entry);
        } catch (...) {
            if (charged) uncharge(entry);
            api_->freeFrame(frame); throw;
        }
        return frame;
    }
    void forget(const VSFrame* frame) noexcept {
        for (auto& slot : inline_) if (slot.frame == frame) { uncharge(slot); slot = {}; return; }
        for (auto& slot : spill_) if (slot.frame == frame) { uncharge(slot); slot = {}; return; }
    }
    const VSAPI* api_;
    std::shared_ptr<ResourceAccount> outputs_, framework_;
    std::array<Entry, 128> inline_{};
    ResourceVector<Entry> spill_;
    struct Mirror { const VSFrame* frame; int plane, stride; ResourceVector<float> pixels; };
    ResourceVector<Mirror> mirrors_;
};

template<auto GetFrame>
const VSFrame* VS_CC checked_frame(int n, int reason, void* data, void** frame_data,
                                   VSFrameContext* context, VSCore* core, const VSAPI* api) noexcept {
    try { return GetFrame(n, reason, data, frame_data, context, core, api); }
    catch (const std::bad_alloc&) { api->setFilterError("nss: resource allocation failed", context); }
    catch (const std::exception& error) { api->setFilterError(error.what(), context); }
    catch (...) { api->setFilterError("nss: unexpected processing failure", context); }
    return nullptr;
}
} // namespace nss
