#pragma once

namespace nss::detail {

// Returns false when the active target has no enabled specialized implementation.
bool dct16_2d_batch_fast(float* patches, int count, bool inverse, bool avx2_enabled = true);

}  // namespace nss::detail
