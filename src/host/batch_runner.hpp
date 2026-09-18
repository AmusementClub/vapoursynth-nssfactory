#pragma once

#include "nss/cpu_batch.hpp"
#include "nss/resources.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace nss::host_detail {

inline constexpr std::size_t kGroupBatchWindow = 32;

template<class Allocator>
inline void append_raster_jobs(std::vector<GroupJob, Allocator>& jobs, int width, int height, int block, int step,
                               const GroupKey& key, int t = 0) {
    if (width < block || height < block || step < 1) {
        return;
    }
    const std::uint64_t first = jobs.size();
    std::uint64_t ordinal = first;
    for (std::int64_t by0 = 0; by0 < static_cast<std::int64_t>(height) - block + step; by0 += step) {
        const int by = static_cast<int>(std::min<std::int64_t>(by0, height - block));
        for (std::int64_t bx0 = 0; bx0 < static_cast<std::int64_t>(width) - block + step; bx0 += step) {
            const int bx = static_cast<int>(std::min<std::int64_t>(bx0, width - block));
            jobs.push_back(GroupJob{ordinal++, bx, by, t, key});
        }
    }
}

// Lazy raster indexing preserves the exact last-row/column and ordinal contract.
class RasterJobs {
public:
    RasterJobs(int w, int h, int b, int step, GroupKey key, int t)
        : w_(w), h_(h), b_(b), step_(step), key_(key), t_(t),
          nx_(checked_int((static_cast<std::uint64_t>(w - b) + step - 1) / step + 1)),
          ny_(checked_int((static_cast<std::uint64_t>(h - b) + step - 1) / step + 1)) {}

    std::size_t size() const { return static_cast<std::size_t>(nx_) * ny_; }
    bool empty() const { return size() == 0; }

    GroupJob operator[](std::size_t i) const {
        return GroupJob{i, static_cast<int>(std::min<std::size_t>((i % nx_) * step_, w_ - b_)),
                        static_cast<int>(std::min<std::size_t>((i / nx_) * step_, h_ - b_)), t_, key_};
    }

private:
    int w_, h_, b_, step_;
    GroupKey key_;
    int t_, nx_, ny_;
};

// The host filters call this after their batch kernel has already bucketed and
// completed every item. At that point a second key sort only creates work that
// the ordered queue must undo, so copy and commit the prepared results in their
// original raster order.
template <typename Result, typename Prepare, typename Commit, typename Jobs>
bool commit_prepared_chunk(const Jobs& jobs, std::size_t begin, std::size_t end, Prepare&& prepare,
                           Commit&& commit) {
    if (begin >= end || end > jobs.size()) {
        return true;
    }
    for (std::size_t index = begin; index < end; ++index) {
        Result result{};
        (void)prepare(jobs[index], result);
        commit(result);
    }
    return true;
}

}  // namespace nss::host_detail
