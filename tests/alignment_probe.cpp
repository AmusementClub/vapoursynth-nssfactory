// Binary FP64 transport for independent Python/SciPy references. This utility
// never computes an expected result with a production finisher.
#include "nss/cpu_twsc_full.hpp"
#include "nss/cpu_nlh_full.hpp"
#include "nss/cpu_nlh.hpp"
#include "nss/cpu_image.hpp"
#include "nss/cpu_api.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {
std::vector<double> read(const char* path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("input open failed");
    const auto bytes = stream.tellg();
    if (bytes < 0 || bytes % 8) throw std::runtime_error("invalid input bytes");
    std::vector<double> data(std::size_t(bytes) / 8); stream.seekg(0);
    stream.read(reinterpret_cast<char*>(data.data()), bytes);
    if (!stream) throw std::runtime_error("input read failed");
    return data;
}
template<class C> void append(std::vector<double>& out, const C& values) { for (auto value : values) out.push_back(double(value)); }
void stats(std::vector<double>& out, const nss::TwscSolverStats& s) {
    out.insert(out.end(), {double(s.iterations), double(s.converged), s.primal, s.coefficient_change, s.auxiliary_change, s.sylvester_residual, double(s.double_svd)});
}
}
int main(int argc, char** argv) {
    try {
        if (argc != 8) throw std::runtime_error("usage: alignment_probe MODE M N R_OR_Q ITERATIONS INPUT OUTPUT");
        const std::string mode = argv[1];
        const int m = std::stoi(argv[2]), n = std::stoi(argv[3]), r = std::stoi(argv[4]), iterations = std::stoi(argv[5]);
        if (m < 1 || n < 1 || m > 4096 || n > 4096) throw std::runtime_error("invalid dimensions");
        auto input = read(argv[6]);
        std::vector<double> out;
        const auto mn = std::size_t(m) * n;
        auto require = [&](std::size_t count) { if (input.size() != count) throw std::runtime_error("unexpected input length"); };
        if (mode == "twsc-fixed") {
            require(mn + std::size_t(m) * r + r + m + n + 3);
            const double* y = input.data(); const double* d = y + mn; const double* s = d + m * r; const double* precision = s + r;
            std::vector<float> sigma(n);
            for (int j = 0; j < n; ++j) sigma[j] = float(precision[m + j]);
            const double* o = precision + m + n;
            nss::TwscSolverOptions options{iterations, o[0], o[1], o[2]};
            nss::TwscWorkspace work; nss::ResourceVector<double> trace; std::vector<double> result(mn);
            const auto status = nss::twsc_solve(y, d, s, m, n, r, precision, sigma.data(), options, result.data(), work, &trace);
            stats(out, status); append(out, result); append(out, trace);
        } else if (mode == "twsc-group") {
            require(mn + m + n + 3);
            std::vector<float> group(mn), rows(m), cols(n), weights(n);
            for (std::size_t i = 0; i < mn; ++i) group[i] = float(input[i]);
            for (int i = 0; i < m; ++i) rows[i] = float(input[mn + i]);
            for (int j = 0; j < n; ++j) cols[j] = float(input[mn + m + j]);
            const double* o = input.data() + mn + m + n;
            nss::TwscWorkspace work;
            const auto status = nss::twsc_filter_full(group.data(), m, n, m, rows.data(), cols.data(), weights.data(), {iterations, o[0], o[1], o[2]}, work);
            stats(out, status); append(out, group); append(out, work.dictionary); append(out, work.singular); append(out, work.mean); append(out, weights);
        } else if (mode == "svd") {
            require(mn);
            std::vector<float> a(input.begin(), input.end());
            nss::TwscWorkspace work; bool fallback = false;
            if (!nss::twsc_svd(a.data(), m, n, m, work, fallback)) throw std::runtime_error("SVD failure");
            out.push_back(double(fallback)); append(out, work.dictionary); append(out, work.singular); append(out, work.vt);
        } else if (mode == "nlh-hard" || mode == "nlh-wiener") {
            require(3 * mn + 3);
            std::vector<float> values(input.begin(), input.begin() + 3 * mn);
            std::vector<int> indices(std::size_t(m) * r);
            nss::pixel_match(values.data() + 2 * mn, m, n, m, r, indices.data());
            nss::NlhGroupOptions options{r, input[3 * mn], mode == "nlh-wiener", input[3 * mn + 1], iterations, input[3 * mn + 2]};
            nss::NlhWorkspace work;
            nss::nlh_filter_full(values.data(), values.data() + mn, m, n, m, options, indices.data(), work);
            append(out, work.numerator); append(out, work.denominator); append(out, indices);
        } else if (mode == "haar-batch") {
            if (m != 4 || n != 16 || iterations < 1 || iterations > 4096)
                throw std::runtime_error("invalid diagnostic Haar batch");
            require(mn * iterations);
            std::vector<float> matrix(input.begin(), input.end());
            for (int i = 0; i < iterations; ++i) nss::nlh_haar2d(matrix.data() + i * mn, m, n, false);
            append(out, matrix);
        } else if (mode == "haar" || mode == "ihaar") {
            require(mn); std::vector<float> matrix(input.begin(), input.end());
            nss::nlh_haar2d(matrix.data(), m, n, mode == "ihaar"); append(out, matrix);
        } else if (mode == "noise") {
            require(2 * mn); nss::ImagePlane plane(m, n), guide(m, n);
            for (std::size_t i = 0; i < mn; ++i) { plane.pixels[i] = float(input[i]); guide.pixels[i] = float(input[mn + i]); }
            out.push_back(nss::nlh_estimate_sigma(plane, guide));
        } else if (mode == "match") {
            if (input.size() < 9) throw std::runtime_error("missing matcher parameters");
            const int frames = int(input[0]), channels = int(input[1]), center = int(input[2]);
            require(9 + mn * frames * channels);
            std::vector<float> images(input.begin() + 9, input.end());
            std::vector<const float*> guides(frames * channels);
            for (int i = 0; i < frames * channels; ++i) guides[i] = images.data() + mn * i;
            nss::ImageSearch search{r, 1, iterations, int(input[3]), int(input[6]), int(input[4]), int(input[5])};
            std::vector<nss::Match> matches(iterations);
            const int count = nss::image_match(guides.data(), frames, channels, m, n, center, int(input[7]), int(input[8]), search, matches.data());
            out.push_back(count);
            for (int i = 0; i < count; ++i) out.insert(out.end(), {double(matches[i].x), double(matches[i].y), double(matches[i].t), double(matches[i].dist)});
        } else throw std::runtime_error("unknown mode");
        std::ofstream stream(argv[7], std::ios::binary); stream.write(reinterpret_cast<const char*>(out.data()), std::streamsize(out.size() * 8));
        if (!stream) throw std::runtime_error("output write failed");
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
