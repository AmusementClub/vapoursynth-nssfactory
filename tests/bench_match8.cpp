#include "cpu/bm/kernel_lab.hpp"
#include "hwy/targets.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <random>
#include <string>
#include <vector>
#if defined(__unix__)
#include <sys/mman.h>
#include <unistd.h>
#endif

using namespace nss;
using namespace nss::detail;
static bool same(const Match* a, const Match* b, int n) {
    for (int i = 0; i < n; ++i)
        if (a[i].x != b[i].x || a[i].y != b[i].y || a[i].t != b[i].t ||
            a[i].ordinal != b[i].ordinal || std::memcmp(&a[i].dist, &b[i].dist, 4)) return false;
    return true;
}
static int candidates(int w, int h, int x, int y, int range) {
    return (std::min(w - 8, x + range) - std::max(0, x - range) + 1) *
           (std::min(h - 8, y + range) - std::max(0, y - range) + 1);
}
struct Query { int x, y; std::vector<MatchReplayItem> items; };

static int verify_guard_pages() {
#if defined(__unix__)
    const std::size_t page = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
    int failed = 0, cases = 0;
    for (int width : {8, 9, 15, 31}) {
        constexpr int height = 17;
        const std::size_t count = width * height;
        const std::size_t pages = (count * sizeof(float) + page - 1) / page;
        void* mapping = mmap(nullptr, (pages + 1) * page, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mapping == MAP_FAILED) return 1;
        char* boundary = static_cast<char*>(mapping) + pages * page;
        if (mprotect(boundary, page, PROT_NONE)) {
            munmap(mapping, (pages + 1) * page);
            return 1;
        }
        float* source = reinterpret_cast<float*>(boundary) - count;
        for (std::size_t i = 0; i < count; ++i) source[i] = float(i % 19) / 19.f;
        for (int range : {0, 1, 7, 20}) {
            Match reference[8];
            const int n = match8_lab_kernel(0)(source, width, width, height,
                                               width - 8, height - 8, range, reference);
            for (int variant : {0, 1, 2}) {
                struct GuardedOutput {
                    std::uint64_t before = 0x123456789abcdef0ULL;
                    Match matches[8];
                    std::uint64_t after = 0x123456789abcdef0ULL;
                } output;
                const int k = match8_lab_kernel(variant)(source, width, width, height,
                                                         width - 8, height - 8, range, output.matches);
                ++cases;
                if (k != n || !same(reference, output.matches, n) ||
                    output.before != 0x123456789abcdef0ULL || output.after != 0x123456789abcdef0ULL) ++failed;
            }
        }
        munmap(mapping, (pages + 1) * page);
    }
    std::printf("{\"verify\":\"match8_guard_pages\",\"cases\":%d,\"failed\":%d}\n", cases, failed);
    return failed;
#else
    std::printf("{\"verify\":\"match8_guard_pages\",\"skipped\":true}\n");
    return 0;
#endif
}

int main(int argc, char** argv) {
    if (const char* target = std::getenv("NSS_LAB_TARGET")) {
        const std::string name(target);
        const auto requested = name == "AVX2" ? HWY_AVX2 : (name == "AVX3" ? HWY_AVX3 : 0);
        if (!requested) return 2;
        if (!(hwy::SupportedTargets() & requested)) return 77;
        hwy::SetSupportedTargetsForTest(requested);
    }
    auto baseline = match8_lab_kernel(0);
    auto capture = match8_capture_kernel();
    if (!baseline || !capture) return 77;
    if (argc == 2 && std::string(argv[1]) == "verify") {
        int failed = verify_guard_pages();
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> noise(-1, 1);
        for (int pattern = 0; pattern < 5; ++pattern) {
            constexpr int w = 31, h = 25, stride = 37;
            std::vector<float> src(stride * h);
            for (auto& value : src) value = pattern == 0 ? noise(rng) : (pattern == 1 ? 0.f : .25f);
            if (pattern == 2) for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) src[y * stride + x] = float((x + y) % 7);
            if (pattern == 3) src[10 * stride + 10] = std::numeric_limits<float>::quiet_NaN();
            if (pattern == 4) src[10 * stride + 10] = std::numeric_limits<float>::infinity();
            for (int range : {0, 1, 7, 20}) for (int x : {0, 9, w - 8}) for (int y : {0, 8, h - 8}) {
                Match ref[8], got[8];
                int n = baseline(src.data(), stride, w, h, x, y, range, ref);
                for (int v = 1; v <= 2; ++v) {
                    int k = match8_lab_kernel(v)(src.data(), stride, w, h, x, y, range, got);
                    if (k != n || !same(ref, got, n)) { ++failed; std::printf("{\"mismatch_variant\":%d,\"pattern\":%d}\n", v, pattern); }
                }
                std::vector<MatchReplayItem> trace(candidates(w, h, x, y, range));
                int k = capture(src.data(), stride, w, h, x, y, range, got, trace.data());
                if (k != n || !same(ref, got, n)) ++failed;
                // Compare all SSDs, including rejected candidates, not only TopK.
                for (int v = 1; v <= 2; ++v) {
                    auto other = trace;
                    match8_capture_kernel(v)(src.data(), stride, w, h, x, y, range, got, other.data());
                    for (std::size_t i = 0; i < trace.size(); ++i)
                        if (trace[i].x != other[i].x || trace[i].y != other[i].y ||
                            std::memcmp(&trace[i].distance, &other[i].distance, 4)) ++failed;
                }
                for (int v = 0; v <= 1; ++v) {
                    k = match8_replay_kernel(v)(trace.data(), trace.size(), w, h, x, y, range, got);
                    if (k != -1 && (k != n || !same(ref, got, n))) ++failed;
                    if (pattern < 3 && k == -1) ++failed;
                }
            }
        }
        std::printf("{\"verify\":\"match8\",\"failed\":%d}\n", failed);
        return failed ? 1 : 0;
    }
    if (argc < 5) { std::fprintf(stderr, "usage: bench_match8 verify | match|replay|replay-insert|replay-reject|replay-ties|capture VARIANT ITERS GRAY8 [TRACE_JSONL]\n"); return 2; }
    const std::string mode(argv[1]);
    const int variant = std::atoi(argv[2]), iterations = std::atoi(argv[3]);
    const bool is_replay = mode.rfind("replay", 0) == 0;
    if (iterations < 1 || variant < 0 || variant > (is_replay ? 1 : 2)) return 2;
    if (mode != "match" && mode != "replay" && mode != "replay-insert" &&
        mode != "replay-reject" && mode != "replay-ties" && mode != "capture") return 2;
    constexpr int w = 1920, h = 1080, range = 7;
    std::ifstream file(argv[4], std::ios::binary);
    std::vector<unsigned char> raw(w * h);
    if (!file.read(reinterpret_cast<char*>(raw.data()), raw.size())) return 2;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> noise(-3.f / 255.f, 3.f / 255.f);
    std::vector<float> src(w * h);
    for (std::size_t i = 0; i < src.size(); ++i) src[i] = raw[i] / 255.f + noise(rng);
    std::vector<Query> queries;
    Match out[8];
    for (int y = 0; y < h - 8; y += 64) for (int x = 0; x < w - 8; x += 64) {
        Query q{x, y, std::vector<MatchReplayItem>(candidates(w, h, x, y, range))};
        const int winners_count = capture(src.data(), w, w, h, x, y, range, out, q.items.data());
        if (mode == "replay-ties") {
            // A synthetic tie control on real coordinates; not real distances.
            for (auto& item : q.items) item.distance = 0.f;
        } else if (mode == "replay-insert" || mode == "replay-reject") {
            std::vector<MatchReplayItem> accepted, rejected;
            std::vector<float> top{0.f};
            for (const auto item : q.items) {
                if (item.x == x && item.y == y) continue;
                if (top.size() < 8 || item.distance < top.back()) {
                    accepted.push_back(item);
                    top.insert(std::upper_bound(top.begin(), top.end(), item.distance), item.distance);
                    if (top.size() > 8) top.pop_back();
                } else rejected.push_back(item);
            }
            if (mode == "replay-insert") q.items = std::move(accepted);
            else {
                // Seed with the seven final winners, in their original raster
                // order. All originally rejected distances remain rejected.
                // Timing includes these seven setup insertions per query.
                std::vector<Match> winners(out, out + winners_count);
                std::sort(winners.begin(), winners.end(), [](const Match& a, const Match& b) { return a.ordinal < b.ordinal; });
                q.items.clear();
                for (const auto& match : winners)
                    if (match.x != x || match.y != y) q.items.push_back({match.x, match.y, match.dist});
                q.items.insert(q.items.end(), rejected.begin(), rejected.end());
            }
            Match replayed[8];
            const int n = match8_replay_kernel(0)(q.items.data(), q.items.size(), w, h, x, y, range, replayed);
            if (n != winners_count || !same(out, replayed, n)) return 1;
        }
        queries.push_back(std::move(q));
    }
    if (mode == "capture") {
        if (argc != 6) return 2;
        std::ofstream trace(argv[5]);
        std::size_t accepted = 0, rejected = 0, ties = 0;
        for (const auto& q : queries) {
            std::vector<float> top{0};
            trace << "{\"x\":" << q.x << ",\"y\":" << q.y << ",\"items\":[";
            bool first = true;
            for (auto item : q.items) {
                std::uint32_t bits; std::memcpy(&bits, &item.distance, 4);
                if (!first) trace << ',';
                first = false;
                trace << '[' << item.x << ',' << item.y << ',' << bits << ']';
                if (item.x == q.x && item.y == q.y) continue;
                ties += std::find(top.begin(), top.end(), item.distance) != top.end();
                if (top.size() < 8 || item.distance < top.back()) {
                    ++accepted; top.insert(std::upper_bound(top.begin(), top.end(), item.distance), item.distance);
                    if (top.size() > 8) top.pop_back();
                } else ++rejected;
            }
            trace << "]}\n";
        }
        if (!trace) return 1;
        std::printf("{\"queries\":%zu,\"accepted\":%zu,\"rejected\":%zu,\"ties\":%zu}\n", queries.size(), accepted, rejected, ties);
        return 0;
    }
    auto matcher = match8_lab_kernel(variant);
    auto replay = is_replay ? match8_replay_kernel(variant) : nullptr;
    std::uint64_t checksum = 0, items = 0;
    for (const auto& q : queries) {
        if (replay) {
            Match reference[8];
            const int expected = match8_replay_kernel(0)(q.items.data(), q.items.size(), w, h, q.x, q.y, range, reference);
            const int actual = replay(q.items.data(), q.items.size(), w, h, q.x, q.y, range, out);
            if (expected < 1 || actual != expected || !same(reference, out, expected)) return 1;
        } else {
            matcher(src.data(), w, w, h, q.x, q.y, range, out);
        }
    }
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        const auto& q = queries[i % queries.size()];
        int n = replay ? replay(q.items.data(), q.items.size(), w, h, q.x, q.y, range, out)
                       : matcher(src.data(), w, w, h, q.x, q.y, range, out);
        if (n <= 0) return 1;
        checksum += out[n - 1].ordinal;
        items += q.items.size();
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::printf("{\"mode\":\"%s\",\"variant\":%d,\"seconds\":%.9f,\"ns_per_query\":%.6f,\"items\":%llu,\"ns_per_item\":%.6f,\"checksum\":%llu}\n",
                mode.c_str(), variant, seconds, seconds * 1e9 / iterations, (unsigned long long)items,
                seconds * 1e9 / items, (unsigned long long)checksum);
}
