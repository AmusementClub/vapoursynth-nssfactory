#pragma once

namespace nss::detail {

// Returns false when the active target has no AVX-512 implementation.
bool dct16_2d_batch_fast(float* patches, int count, bool inverse);

}  // namespace nss::detail
