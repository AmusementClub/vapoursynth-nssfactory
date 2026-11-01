// SPDX-License-Identifier: GPL-2.0-only
#pragma once

#include "vulkan/runtime/slot.hpp"

#include <chrono>
#include <cstdio>
#include <cstdint>
#include <mutex>
#include <string>

namespace nss_vk {

struct TimingValues {
    double upload_ns = 0.0, match_ns = 0.0, filter_ns = 0.0, aggregate_ns = 0.0, download_ns = 0.0, wait_ns = 0.0;
    std::uint64_t submit_count = 0;
};

class TimingStats {
public:
    void add(const TimingValues& values) {
        std::lock_guard lock(mutex_);
        upload_ns_ += values.upload_ns;
        match_ns_ += values.match_ns;
        filter_ns_ += values.filter_ns;
        aggregate_ns_ += values.aggregate_ns;
        download_ns_ += values.download_ns;
        wait_ns_ += values.wait_ns;
        submit_count_ += values.submit_count;
    }

    void print(const std::string& filter) const {
        std::lock_guard lock(mutex_);
        std::fprintf(stderr,
                     "%s timings: {\"_NSSVkUploadNs\":%lld,\"_NSSVkMatchNs\":%lld,\"_NSSVkFilterNs\":%lld,"
                     "\"_NSSVkAggregateNs\":%lld,\"_NSSVkDownloadNs\":%lld,\"_NSSVkWaitNs\":%lld,"
                     "\"_NSSVkSubmitCount\":%llu}\n",
                     filter.c_str(), static_cast<long long>(upload_ns_), static_cast<long long>(match_ns_),
                     static_cast<long long>(filter_ns_), static_cast<long long>(aggregate_ns_),
                     static_cast<long long>(download_ns_), static_cast<long long>(wait_ns_),
                     static_cast<unsigned long long>(submit_count_));
    }

private:
    mutable std::mutex mutex_;
    double upload_ns_ = 0.0, match_ns_ = 0.0, filter_ns_ = 0.0, aggregate_ns_ = 0.0, download_ns_ = 0.0, wait_ns_ = 0.0;
    std::uint64_t submit_count_ = 0;
};

class FrameTimer {
public:
    explicit FrameTimer(Slot& slot) : slot_(slot), enabled_(slot.has_timestamps()) {
        if (!enabled_) return;
        slot_.mark(Slot::Upload, 0);
        slot_.mark(Slot::Compute, 2);
    }
    void upload_end() { mark(Slot::Upload, 1); }
    void match_end() { mark(Slot::Compute, stage_query(0)); }
    void filter_end() { mark(Slot::Compute, stage_query(1)); }
    void aggregate_end() { mark(Slot::Compute, stage_query(2)); }
    void download_begin() { mark(Slot::Download, 3); }
    void download_end() { mark(Slot::Download, 4); }
    void next_stage() {
        if (stage_ + 1 < kMaxStages) ++stage_;
    }

    void wait() {
        if (!enabled_) {
            slot_.wait();
            return;
        }
        const auto begin = std::chrono::steady_clock::now();
        slot_.wait();
        values_.wait_ns += std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - begin).count();
        const auto& times = slot_.timestamps_ns();
        if (times.size() != kTimingQueryCount) return;
        values_.upload_ns += duration(times[0], times[1]);
        for (std::uint32_t stage = 0; stage <= stage_; ++stage) {
            values_.match_ns += duration(times[2], times[stage_query(0, stage)]);
            values_.filter_ns += duration(times[stage_query(0, stage)], times[stage_query(1, stage)]);
            values_.aggregate_ns += duration(times[stage_query(1, stage)], times[stage_query(2, stage)]);
        }
        values_.download_ns += duration(times[3], times[4]);
        values_.submit_count += slot_.submit_count();
    }

    const TimingValues& values() const noexcept { return values_; }

private:
    static constexpr std::uint32_t kMaxStages = (kTimingQueryCount - 5) / 3;
    static double duration(double begin, double end) { return end >= begin ? end - begin : 0.0; }
    std::uint32_t stage_query(std::uint32_t part, std::uint32_t stage) const { return 5 + 3 * stage + part; }
    std::uint32_t stage_query(std::uint32_t part) const { return stage_query(part, stage_); }
    void mark(Slot::Stage stage, std::uint32_t query) {
        if (enabled_) slot_.mark(stage, query);
    }

    Slot& slot_;
    bool enabled_ = false;
    TimingValues values_;
    std::uint32_t stage_ = 0;
};

}  // namespace nss_vk
