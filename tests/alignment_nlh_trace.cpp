// Test-only production-primitive replay. This is diagnostic, never an oracle.
// Dump fixed RGB spatial defaults and selectors to localize full-image errors.
#include "nss/cpu_image.hpp"
#include "nss/cpu_nlh.hpp"
#include "nss/cpu_nlh_full.hpp"
#include "cpu/common/image_internal.hpp"
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace {
void save(const std::filesystem::path& p, const nss::ImageFrame& f) {
    std::ofstream out(p, std::ios::binary);
    for (const auto& plane : f.planes)
        out.write(reinterpret_cast<const char*>(plane.pixels.data()), plane.pixels.size() * sizeof(float));
    if (!out) throw std::runtime_error("trace write failed");
}
}
int main(int argc, char** argv) {
    try {
        if (argc != 6) throw std::runtime_error("usage: alignment_nlh_trace WIDTH HEIGHT SIGMA RGB_F32 OUTDIR");
        const int width = std::stoi(argv[1]), height = std::stoi(argv[2]);
        const float sigma = std::stof(argv[3]) / 255.f;
        constexpr int block = 7, m = 49, n = 16, q = 4;
        if (width < 10 || height < 10 || width > 4096 || height > 4096 || !(sigma > 0))
            throw std::runtime_error("invalid diagnostic shape/noise");
        const std::filesystem::path root(argv[5]);
        if (!std::filesystem::create_directory(root)) throw std::runtime_error("trace directory already exists");
        nss::ImageSequence input(1);
        std::ifstream file(argv[4], std::ios::binary);
        for (int c = 0; c < 3; ++c) {
            auto& plane = input[0].planes[c]; plane = nss::ImagePlane(width, height);
            file.read(reinterpret_cast<char*>(plane.pixels.data()), plane.pixels.size() * sizeof(float));
            input[0].sigma[c] = sigma;
        }
        if (!file || file.peek() != std::char_traits<char>::eof()) throw std::runtime_error("invalid input file");
        nss::nlh_rgb_to_yuv(input[0], true);
        save(root / "working.f32", input[0]);
        std::ofstream noise_file(root / "noise.f32", std::ios::binary);
        noise_file.write(reinterpret_cast<const char*>(input[0].sigma.data()), 3 * sizeof(float));
        nss::ImageSequence basic = input;
        std::array<nss::NlhWorkspace, 3> work;
        std::array<nss::Match, n> matches;
        std::array<int, m * q> indices;
        std::array<float, m * n> guide_group;
        std::array<float, 3 * m * n> pixels, refs;
        nss::ImageSearch search{block, 1, n, 40, 0, 2, 4};
        for (int stage = 0; stage < 3; ++stage) {
            const bool wiener = stage == 2;
            auto data = wiener ? input : basic;
            if (!wiener) for (int c = 0; c < 3; ++c) for (std::size_t i = 0; i < data[0].planes[c].pixels.size(); ++i)
                data[0].planes[c].pixels[i] = nss::image_detail::finite_float(.6 * basic[0].planes[c].pixels[i] + .4 * input[0].planes[c].pixels[i]);
            const auto& guide = wiener ? basic : data;
            save(root / ("data" + std::to_string(stage) + ".f32"), data[0]);
            save(root / ("basic" + std::to_string(stage) + ".f32"), basic[0]);
            std::ofstream blocks(root / ("blocks" + std::to_string(stage) + ".u32"), std::ios::binary);
            std::ofstream rows(root / ("pixels" + std::to_string(stage) + ".u8"), std::ios::binary);
            auto accum = nss::image_detail::accumulator(input, 3);
            const float* luma = guide[0].planes[0].pixels.data();
            nss::image_detail::raster(width, height, block, 1, [&](int x, int y) {
                if (nss::image_match(&luma, 1, 1, width, height, 0, x, y, search, matches.data()) != n)
                    throw std::runtime_error("trace requires full groups");
                std::array<std::uint32_t, n> packed_matches;
                for (int j = 0; j < n; ++j) {
                    const auto& hit = matches[j];
                    packed_matches[j] = hit.y * (width - block + 1) + hit.x;
                    nss::pack_patch(guide_group.data() + j * m, m, luma, width, hit.x, hit.y, block, width, height);
                }
                blocks.write(reinterpret_cast<const char*>(packed_matches.data()), sizeof(packed_matches));
                nss::pixel_match(guide_group.data(), m, n, m, q, indices.data());
                std::array<std::uint8_t, m * q> packed_rows;
                for (int i = 0; i < m * q; ++i) packed_rows[i] = std::uint8_t(indices[i]);
                rows.write(reinterpret_cast<const char*>(packed_rows.data()), sizeof(packed_rows));
                std::array<nss::NlhGroupOptions, 3> config;
                std::array<nss::NlhFullBatchItem, 3> batch;
                for (int c = 0; c < 3; ++c) {
                    for (int j = 0; j < n; ++j) {
                        const auto& hit = matches[j];
                        nss::pack_patch(pixels.data() + c * m * n + j * m, m, data[0].planes[c].pixels.data(), width, hit.x, hit.y, block, width, height);
                        nss::pack_patch(refs.data() + c * m * n + j * m, m, basic[0].planes[c].pixels.data(), width, hit.x, hit.y, block, width, height);
                    }
                    config[c] = {q, input[0].sigma[c], wiener, 1, 2, nss::kNlhWienerSigmaScale};
                    batch[c] = {pixels.data() + c * m * n, wiener ? refs.data() + c * m * n : nullptr, m, n, m, &config[c], indices.data(), &work[c]};
                }
                nss::nlh_filter_full_batch(batch.data(), 3);
                for (int c = 0; c < 3; ++c) for (int j = 0; j < n; ++j)
                    nss::image_detail::add_pixel_matrix(accum[0].planes[c], matches[j].x, matches[j].y, block,
                        work[c].numerator.data() + j * m, work[c].denominator.data() + j * m);
            });
            if (!blocks || !rows) throw std::runtime_error("selector trace write failed");
            if (!wiener) {
                basic = nss::image_detail::finish(accum, input, 3);
                save(root / ("result" + std::to_string(stage) + ".f32"), basic[0]);
            } else {
                auto output = nss::image_detail::contributions(accum, input, 3, {});
                nss::nlh_yuv_to_rgb(output.numerator[0]);
                for (int c = 0; c < 3; ++c) for (std::size_t i = 0; i < output.numerator[0].planes[c].pixels.size(); ++i)
                    output.numerator[0].planes[c].pixels[i] = float(double(output.numerator[0].planes[c].pixels[i]) / output.denominator[0].planes[c].pixels[i]);
                save(root / "output.f32", output.numerator[0]);
            }
            std::cout << "stage " << stage << " complete\n" << std::flush;
        }
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
