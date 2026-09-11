#pragma once
#include "nss/params.hpp"
#include <string>

// Single high-quality clustered-PCA model. No algorithm-mode selector.
// All algorithm state is invocation-local. Large dynamic buffers use the
// caller's ResourceScope; small metadata/diagnostics use ordinary storage.
namespace nss::ncsr_hq {
void denoise(const float* input, int width, int height, int stride, float* output,
             int output_stride, SearchConfig config, float sigma, int iterations,
             float delta, std::string& trace);
}
