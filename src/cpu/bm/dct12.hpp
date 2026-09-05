#pragma once

namespace nss::detail {

// Returns false when the active target has no AVX2 implementation.
bool dct12_2d_batch_fast(float* patches, int count, bool inverse);

}  // namespace nss::detail
