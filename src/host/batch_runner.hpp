#pragma once

#include "nss/cpu_batch.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace nss::host_detail {

inline constexpr std::size_t kGroupBatchWindow = 32;

inline void append_raster_jobs(std::vector<GroupJob>& jobs, int width, int height, int block, int step,
                               const GroupKey& key, int t = 0) {
    if (width < block || height < block || step < 1) {
        return;
    }
    const std::uint64_t first = jobs.size();
    std::uint64_t ordinal = first;
    for (int by0 = 0; by0 < height - block + step; by0 += step) {
        const int by = std::min(by0, height - block);
        for (int bx0 = 0; bx0 < width - block + step; bx0 += step) {
            const int bx = std::min(bx0, width - block);
            jobs.push_back(GroupJob{ordinal++, bx, by, t, key});
        }
    }
}

// Execute one bounded raster chunk. `prepare` fills a result for a job in any
// bucket order and returns false for a skipped/failed group. `commit` receives
// results strictly in ordinal order, including failed results so gaps cannot
// stall the queue.
template <typename Result, typename Prepare, typename Commit>
bool execute_ordered_chunk(const std::vector<GroupJob>& jobs, std::size_t begin, std::size_t end, Prepare&& prepare,
                           Commit&& commit) {
    if (begin >= end || end > jobs.size()) {
        return true;
    }
    std::vector<std::size_t> order;
    order.reserve(end - begin);
    for (std::size_t i = begin; i < end; ++i) {
        order.push_back(i);
    }
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        if (jobs[a].key == jobs[b].key) {
            return jobs[a].ordinal < jobs[b].ordinal;
        }
        return jobs[a].key < jobs[b].key;
    });

    OrderedCommitQueue<Result> queue(jobs[begin].ordinal, kGroupBatchWindow);
    for (const std::size_t index : order) {
        Result result{};
        (void)prepare(jobs[index], result);
        if (!queue.push(jobs[index].ordinal, result)) {
            // A producer that exceeds the reorder window must make progress
            // before retrying; a missing ordinal is a hard caller error.
            if (queue.drain(commit) == 0 || !queue.push(jobs[index].ordinal, result)) {
                return false;
            }
        }
        queue.drain(commit);
    }
    queue.finish(commit);
    return queue.complete();
}

template <typename Result, typename Prepare, typename Commit>
bool execute_ordered_jobs(const std::vector<GroupJob>& jobs, Prepare&& prepare, Commit&& commit) {
    for (std::size_t begin = 0; begin < jobs.size(); begin += kGroupBatchWindow) {
        const std::size_t end = std::min(jobs.size(), begin + kGroupBatchWindow);
        if (!execute_ordered_chunk<Result>(jobs, begin, end, prepare, commit)) {
            return false;
        }
    }
    return true;
}

}  // namespace nss::host_detail
