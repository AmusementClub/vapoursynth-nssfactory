#pragma once
#include "nss/params.hpp"
#include <string>

// Research-only hooks in the isolated experiment source. No production API.
namespace nss::alignment_lab {
enum Flags { Statistics = 1, Weights = 2, TargetOnly = 4, Dictionary = 8, Reuse = 16 };
bool finish_codes(float* codes, int dimensions, int columns, float sigma,
                  const float* distances, int patch_dimensions);
void denoise(const float* input, int width, int height, int stride, float* output,
             int output_stride, SearchConfig config, float sigma, int iterations,
             float delta, int flags, std::string& trace);
bool self_test();
}
