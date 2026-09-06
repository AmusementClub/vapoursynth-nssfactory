#include "nss/cpu_api.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {

struct Candidate {
    int x;
    int y;
    float dist;
    std::uint32_t ordinal;
};

bool candidate_less(const Candidate& a, const Candidate& b) {
    if (a.dist != b.dist) {
        return a.dist < b.dist;
    }
    return a.ordinal < b.ordinal;
}

std::vector<Candidate> exact_group(const float* frame, int stride, int width, int height, int bx, int by, int block,
                                   int range, int group) {
    const int max_x = width - block;
    const int max_y = height - block;
    const int cx = std::clamp(bx, 0, max_x);
    const int cy = std::clamp(by, 0, max_y);
    const int left = std::max(0, cx - range);
    const int right = std::min(max_x, cx + range);
    const int top = std::max(0, cy - range);
    const int bottom = std::min(max_y, cy + range);
    const float* query = frame + cy * stride + cx;
    std::vector<Candidate> candidates;
    candidates.reserve(static_cast<std::size_t>((right - left + 1) * (bottom - top + 1) - 1));
    const int span = right - left + 1;
    for (int y = top; y <= bottom; ++y) {
        for (int x = left; x <= right; ++x) {
            if (x == cx && y == cy) {
                continue;
            }
            const auto ordinal = static_cast<std::uint32_t>((y - top) * span + (x - left) + 1);
            candidates.push_back(
                Candidate{x, y, nss::ssd_block(query, stride, frame + y * stride + x, stride, block), ordinal});
        }
    }
    std::sort(candidates.begin(), candidates.end(), candidate_less);
    if (static_cast<int>(candidates.size()) > group - 1) {
        candidates.resize(static_cast<std::size_t>(group - 1));
    }
    return candidates;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 7) {
        std::fprintf(stderr, "usage: %s GRAY8 BLOCK STEP GROUP RANGE MAX_QUERIES\n", argv[0]);
        return 2;
    }
    const std::string path = argv[1];
    const int block = std::atoi(argv[2]);
    const int step = std::atoi(argv[3]);
    const int group = std::atoi(argv[4]);
    const int range = std::atoi(argv[5]);
    const int max_queries = std::max(1, std::atoi(argv[6]));
    constexpr int width = 1920;
    constexpr int height = 1080;
    std::ifstream input(path, std::ios::binary);
    std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    if (bytes.size() != static_cast<std::size_t>(width * height)) {
        std::fprintf(stderr, "invalid gray8 sample: %s (%zu bytes)\n", path.c_str(), bytes.size());
        return 2;
    }
    std::vector<float> frame(bytes.size());
    std::uint32_t state = 42u;
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        state = state * 1664525u + 1013904223u;
        const float noise = (static_cast<float>(state >> 8) * (1.f / 16777216.f) - 0.5f) * (6.f / 255.f);
        frame[i] = static_cast<float>(bytes[i]) * (1.f / 255.f) + noise;
    }

    const int nx = (width - block + step - 1) / step;
    const int ny = (height - block + step - 1) / step;
    const int total = nx * ny;
    const int query_stride = std::max(1, (total + max_queries - 1) / max_queries);
    std::uint64_t exact_members = 0;
    std::uint64_t retained_members = 0;
    std::uint64_t exact_groups = 0;
    std::uint64_t queries = 0;
    double exact_distance_sum = 0.0;
    double candidate_distance_sum = 0.0;
    for (int ordinal = 0; ordinal < total; ordinal += query_stride) {
        const int ix = ordinal % nx;
        const int iy = ordinal / nx;
        const int bx = std::min(ix * step, width - block);
        const int by = std::min(iy * step, height - block);
        const auto exact = exact_group(frame.data(), width, width, height, bx, by, block, range, group);
        nss::Match got[nss::kBmMaxGroup]{};
        const int count = nss::spatial_match(frame.data(), width, width, height, bx, by, block, range, group, got);
        bool same = count == static_cast<int>(exact.size()) + 1;
        for (const auto& want : exact) {
            ++exact_members;
            exact_distance_sum += want.dist;
            bool found = false;
            for (int i = 1; i < count; ++i) {
                if (got[i].x == want.x && got[i].y == want.y) {
                    found = true;
                    break;
                }
            }
            retained_members += found ? 1u : 0u;
        }
        for (int i = 1; i < count; ++i) {
            candidate_distance_sum += got[i].dist;
            if (i - 1 >= static_cast<int>(exact.size()) || got[i].x != exact[static_cast<std::size_t>(i - 1)].x ||
                got[i].y != exact[static_cast<std::size_t>(i - 1)].y) {
                same = false;
            }
        }
        exact_groups += same ? 1u : 0u;
        ++queries;
    }
    const double recall = exact_members == 0 ? 1.0 : static_cast<double>(retained_members) / exact_members;
    const double exact_group_rate = queries == 0 ? 1.0 : static_cast<double>(exact_groups) / queries;
    const double distance_ratio = exact_distance_sum == 0.0 ? 1.0 : candidate_distance_sum / exact_distance_sum;
    const char* shortlist = std::getenv("NSS_BM_APPROX_SHORTLIST");
    std::printf(
        "{\"schema\":\"nssfactory.bm_match_quality.v1\",\"sample\":\"%s\",\"block\":%d,"
        "\"step\":%d,\"group\":%d,\"range\":%d,\"shortlist\":%s,\"queries\":%llu,"
        "\"member_recall\":%.9f,\"exact_group_rate\":%.9f,\"distance_sum_ratio\":%.9f}\n",
        path.c_str(), block, step, group, range, shortlist && *shortlist ? shortlist : "0",
        static_cast<unsigned long long>(queries), recall, exact_group_rate, distance_ratio);
    return 0;
}
