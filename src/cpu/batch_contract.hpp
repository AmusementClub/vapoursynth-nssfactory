#pragma once

#include "nss/params.hpp"

#include <algorithm>

namespace nss::detail {

// Batch APIs report the smallest failed caller index, even when execution is
// reordered into shape buckets.
inline void record_batch_failure(int original_index, int& first_error) noexcept {
    const int code = original_index + 1;
    if (first_error == 0 || code < first_error) {
        first_error = code;
    }
}

// Every matcher writes the reference match plus up to kBmMaxGroup - 1
// candidates. Validate the caller-provided leading dimension before entering
// a kernel so a malformed direct CPU call cannot write into the next item.
inline bool match_capacity_valid(int group, int match_stride) noexcept {
    return group > 0 && match_stride >= std::min(group, kBmMaxGroup);
}

}  // namespace nss::detail
