#include "lssc_trace.hpp"
#include "nss/cpu_api.hpp"
#include "nss/cpu_lssc.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <vector>

namespace {
std::ofstream trace;
int frame_id = 0, call_id = -1;
template<class T> void save(const std::filesystem::path& path, const std::vector<T>& a) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(a.data()), a.size() * sizeof(T));
    if (!out) throw std::runtime_error("write failed");
}
template<class T> void list(const T* p, int n) {
    trace << '[';
    for (int i = 0; i < n; ++i) { if (i) trace << ','; trace << +p[i]; }
    trace << ']';
}
}

void nss_trace_omp(const float* y, int m, const float* dictionary, int atoms, int ldd,
                   const float* residual, const float* correlations, const char* used,
                   int selected, float magnitude, int step) {
    if (step == 0) ++call_id;
    std::vector<double> oracle(atoms);
    int oracle_best = -1;
    double maximum = 0, second = 0;
    for (int a = 0; a < atoms; ++a) {
        // Independent FP64 dot over the exact FP32 residual and dictionary.
        double sum = 0;
        for (int i = 0; i < m; ++i) sum += double(dictionary[a * ldd + i]) * double(residual[i]);
        oracle[a] = sum;
        if (used[a]) continue;
        const double value = std::abs(sum);
        if (value > maximum) { second = maximum; maximum = value; oracle_best = a; }
        else if (value > second) second = value;
    }
    trace << "{\"frame\":" << frame_id << ",\"call\":" << call_id << ",\"step\":" << step
          << ",\"selected\":" << selected << ",\"magnitude\":" << magnitude
          << ",\"oracle_selected\":" << oracle_best << ",\"oracle_margin\":" << maximum - second
          << ",\"correlations\":";
    list(correlations, atoms); trace << ",\"oracle_correlations\":"; list(oracle.data(), atoms);
    trace << ",\"used\":"; list(used, atoms);
    trace << ",\"residual\":"; list(residual, m);
    trace << ",\"input\":"; list(y, m);
    trace << "}\n";
}

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const std::filesystem::path out(argv[1]);
    if (!std::filesystem::create_directory(out)) return 3;
    trace.open(out / "omp.jsonl"); trace << std::setprecision(17);
    constexpr int w = 36, h = 34, block = 8, m = block * block;
    const int n = nss::lssc_grid_count(w, h, block, block);
    const int atoms = std::min(n, nss::kLsscDefaultAtoms);
    for (frame_id = 0; frame_id < 3; ++frame_id) {
        const auto dir = out / std::to_string(frame_id); std::filesystem::create_directory(dir);
        std::vector<float> source(w * h), num(w * h), den(w * h), result(w * h);
        for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x)
            source[y * w + x] = float((x * 3 + y * 5 + frame_id * 7) % 97) / 128.f;
        std::vector<float> work(nss::lssc_denoise_work_floats(w, h, block, block));
        nss::lssc_denoise_plane(source.data(), w, h, w, num.data(), den.data(), w,
                                block, block, 3.f / 255.f, work.data(), int(work.size()));
        nss::aggregate_finish(result.data(), num.data(), den.data(), source.data(), w, h, w, w, w);
        save(dir / "output.f32", result); save(dir / "den.f32", den);
        save(dir / "dictionary.f32", std::vector<float>(work.begin() + n * m, work.begin() + n * m + m * atoms));
    }
    return trace.good() ? 0 : 4;
}
