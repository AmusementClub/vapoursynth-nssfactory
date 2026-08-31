#pragma once

namespace nss {

struct PlaneView {
    const float* ptr = nullptr;
    float* mut = nullptr;
    int width = 0;
    int height = 0;
    int stride = 0;
};

}  // namespace nss
