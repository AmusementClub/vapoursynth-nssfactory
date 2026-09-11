#include "nss/cpu_image.hpp"
#include <iostream>
#include <stdexcept>

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
template<class F> void rejects(F call) {
    try { call(); }
    catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("invalid explicit option was accepted");
}
nss::ImageFrame metadata(int width, int height, float sigma) {
    nss::ImageFrame frame;
    for (int c = 0; c < 3; ++c) {
        frame.planes[c].width = width;
        frame.planes[c].height = height;
        frame.sigma[c] = sigma;
        frame.sigma_units[c] = double(sigma) * 255;
    }
    return frame;
}
}

int main() {
    try {
        nss::ImageSequence input(1, metadata(32, 24, .1f));
        nss::NlhImageOptions explicit_options;
        explicit_options.block = {6, 9}; explicit_options.step = {3, 7};
        explicit_options.group = {8, 32}; explicit_options.q = {2, 8};
        explicit_options.window = {20, 129}; explicit_options.basic_iterations = 3;
        explicit_options.wiener_iterations = 1; explicit_options.basic_mix = 0;
        explicit_options.hard_strength = 0; explicit_options.wiener_sigma_scale = 0;
        const auto exact = nss::nlh_resolve_options(input, 3, 0, explicit_options);
        require(exact.block == explicit_options.block && exact.step == explicit_options.step &&
                exact.group == explicit_options.group && exact.q == explicit_options.q &&
                exact.window == explicit_options.window && exact.basic_iterations == 3 &&
                exact.wiener_iterations == 1 && exact.basic_mix == 0 &&
                exact.hard_strength == 0 && exact.wiener_sigma_scale == 0,
                "explicit fields, including zero coefficients, must override the preset");
        input[0] = metadata(2, 3, .1f);
        const auto small = nss::nlh_resolve_options(input, 3, 0, {});
        for (int s = 0; s < 2; ++s)
            require(small.block[s] == 2 && small.step[s] <= 2 && small.q[s] >= 2 && small.q[s] <= 4,
                    "automatic block/step/q must fit a legal small image");
        auto invalid = explicit_options;
        rejects([&] { nss::nlh_resolve_options(input, 3, 0, invalid); });
        invalid = {}; invalid.block = {2, 2}; invalid.q = {8, 8};
        rejects([&] { nss::nlh_resolve_options(input, 3, 0, invalid); });
        invalid = {}; invalid.step = {16, 16};
        rejects([&] { nss::nlh_resolve_options(input, 3, 0, invalid); });
        invalid = {}; invalid.hard_strength = -2;
        rejects([&] { nss::nlh_resolve_options(input, 3, 0, invalid); });
        invalid = {}; invalid.group = {2, 2}; invalid.ps_num = 3;
        rejects([&] { nss::nlh_resolve_options(input, 3, 0, invalid); });
        input[0] = metadata(1, 1, .1f);
        rejects([&] { nss::nlh_resolve_options(input, 3, 0, {}); });
        input[0] = metadata(1, 1, 0);
        const auto bypass = nss::nlh_resolve_options(input, 3, 0, {});
        require(bypass.block[0] >= 2 && bypass.block[1] >= 2, "zero-noise identity may use a nominal preset");

        input.assign(3, metadata(32, 24, .1f));
        for (auto& frame : input) for (int c = 1; c < 3; ++c) {
            frame.planes[c].width = 8; frame.planes[c].height = 8;
        }
        input[1].sigma.fill(0); input[1].sigma_units.fill(0);
        invalid = {}; invalid.block = {16, 16};
        rejects([&] { nss::nlh_resolve_options(input, 3, 1, invalid); });
        const auto center_only = nss::nlh_resolve_options({&input[1], 1}, 3, 0, invalid);
        require(center_only.block[0] == 16, "a bypassed center must not inspect neighbors for geometry");
        for (auto& frame : input) for (int c = 1; c < 3; ++c) {
            frame.sigma[c] = 0; frame.sigma_units[c] = 0;
        }
        const auto luma_only = nss::nlh_resolve_options(input, 3, 0, invalid);
        require(luma_only.block[0] == 16, "disabled chroma must not reduce the active luma block");
        std::cout << "NLH resolution: explicit overrides, zero coefficients, small images and temporal geometry PASS\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
