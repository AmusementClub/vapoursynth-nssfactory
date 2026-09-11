#include "nss/cpu_image.hpp"
#include "nss/backend.hpp"
#include "hwy/highway.h"
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string_view>

int main(int argc, char** argv) {
    if (argc > 1) {
        const auto target = std::string_view(argv[1]) == "avx2" ? HWY_AVX2 :
                            std::string_view(argv[1]) == "avx3" ? HWY_AVX3 : 0;
        const auto caps = nss::backend_caps();
        if (!target || !(caps.compiled_targets & caps.runtime_targets & target)) return 77;
        hwy::SetSupportedTargetsForTest(target);
    }
    try {
        for (int layout = 0; layout < 4; ++layout) for (bool external : {false, true})
        for (int pattern = 0; pattern < 3; ++pattern) {
            const int planes = layout == 0 ? 1 : 3;
            nss::ImageFrame frame, reference;
            for (int c = 0; c < planes; ++c) {
                const int width = layout >= 2 ? (c ? 9 : 18) : 19;
                const int height = layout == 3 ? (c ? 8 : 16) : 17;
                frame.planes[c] = nss::ImagePlane(width, height);
                reference.planes[c] = nss::ImagePlane(width, height);
                for (std::size_t i = 0; i < frame.planes[c].pixels.size(); ++i) {
                    frame.planes[c].pixels[i] = pattern == 0 ? .375f :
                        pattern == 1 ? float((i * 37 + 13 * c) % 257) / 256 :
                        float(std::sin((i + c * 41) * .137) * std::ldexp(1., int(i % 9) - 4));
                    reference.planes[c].pixels[i] = float((i * 23 + 71 * c) % 193) / 192;
                }
            }
            const auto& guide = external ? reference : frame;
            std::array<float, 3> expected{};
            for (int c = 0; c < planes; ++c) {
                const auto& channel = frame.planes[c];
                const auto& luma = guide.planes[0];
                if (channel.width == luma.width && channel.height == luma.height)
                    expected[c] = nss::nlh_estimate_sigma(channel, luma);
                else {
                    const auto resized = nss::image_area_guide(luma, channel.width, channel.height);
                    expected[c] = nss::nlh_estimate_sigma(channel, resized);
                }
            }
            const auto original = frame;
            nss::nlh_estimate_frame_sigma(frame, guide, planes);
            for (int c = 0; c < planes; ++c) {
                if (std::memcmp(&frame.sigma[c], &expected[c], sizeof(float)) ||
                    frame.sigma_units[c] != double(expected[c]) * 255 ||
                    frame.planes[c].pixels != original.planes[c].pixels)
                    throw std::runtime_error("shared NLH sigma differs from independent channel reductions");
            }
        }
        std::cout << "shared NLH noise: exact Gray/RGB/YUV geometry, guide, ties and reduction order PASS\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
