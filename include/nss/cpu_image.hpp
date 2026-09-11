#pragma once

#include "nss/cpu_twsc_full.hpp"
#include "nss/params.hpp"
#include <array>
#include <span>

namespace nss {

struct ImagePlane {
    int width = 0, height = 0;
    ResourceVector<float> pixels;
    ImagePlane() = default;
    ImagePlane(int w, int h) : width(w), height(h), pixels(checked_product({std::size_t(w), std::size_t(h)})) {}
};
struct ImageFrame {
    std::array<ImagePlane, 3> planes;
    std::array<float, 3> sigma{}; // normalized standard deviation in this frame's working color domain
    std::array<double, 3> sigma_units{-1, -1, -1}; // exact public/preset units when known
};
inline double image_sigma_units(const ImageFrame& frame, int channel) {
    return frame.sigma_units[channel] >= 0 ? frame.sigma_units[channel] : double(frame.sigma[channel]) * 255;
}
using ImageSequence = ResourceVector<ImageFrame>;

struct ImageSearch {
    int block = 8, step = 1, group = 16, window = 40;
    int radius = 0, ps_num = 2, ps_range = 4;
};
struct TwscImageOptions {
    int block = 0, group = 0, iterations = 0;
    int step = 1, window = 60, radius = 0, ps_num = 2, ps_range = 4;
    double lambda2 = 1, delta = 0;
    TwscSolverOptions solver;
};
struct NlhImageOptions {
    // Zero integer fields and exactly -1 coefficient fields request a preset.
    // Public parsing permits these sentinels only through omitted arguments.
    std::array<int, 2> block{0, 0}, step{0, 0}, group{0, 0}, q{0, 0}, window{0, 0};
    int basic_iterations = 0, wiener_iterations = 0;
    int radius = 0, ps_num = 2, ps_range = 4;
    double basic_mix = -1, hard_strength = -1, wiener_sigma_scale = -1;
    bool real_noise = false;
};
struct ImageFilterStats {
    std::uint64_t groups = 0, max_iteration_groups = 0, double_svd_groups = 0;
    double max_sylvester_residual = 0;
};
struct ImageContributions {
    ImageSequence numerator, denominator;
    ImageFilterStats stats;
};

// Exact-size windows, deterministic self-first Top-K; guides are frame-major,
// with nch equally-sized planes per frame. Public Match.t is a sequence index.
struct Match;
int image_match(const float* const* guides, int frames, int nch, int width, int height,
                int t, int x, int y, const ImageSearch& search, Match* matches);

// Paper equations (3)-(4), fixed 8x8 / 16 / q4 / W40 / step1 bootstrap.
// Distances/selectors use guide; channel intensities determine its noise value.
float nlh_estimate_sigma(const ImagePlane& channel, const ImagePlane& guide);
// NLH channels of equal geometry share luma matching and pixel indices. Keep
// each channel's distance and group reductions in the single-channel order.
void nlh_estimate_frame_sigma(ImageFrame& frame, const ImageFrame& guide, int planes);

// Shared by execution and host diagnostics. Only omitted block/step/q values
// adapt to the active planes' geometry; explicit invalid shapes remain errors.
NlhImageOptions nlh_resolve_options(std::span<const ImageFrame> input, int planes, int center,
                                   const NlhImageOptions& requested);

// All supplied frames are genuine members of the fixed request window. Every
// intermediate iteration is synchronous over all reference frames. Only center
// queries contribute to the returned final numerator/denominator sequence.
ImageContributions twsc_image(const ImageSequence& input, const ImageSequence* reference,
                              int planes, int center, const TwscImageOptions& options);
ImageContributions nlh_image(const ImageSequence& input, const ImageSequence* reference,
                             int planes, int center, const NlhImageOptions& options);

// BT.601 full-range RGB <-> YCbCr (chroma centered at zero). No sample clipping.
void nlh_rgb_to_yuv(ImageFrame& frame, bool propagate_sigma);
void nlh_yuv_to_rgb(ImageFrame& frame);
ImagePlane image_area_guide(const ImagePlane& luma, int width, int height);

} // namespace nss
