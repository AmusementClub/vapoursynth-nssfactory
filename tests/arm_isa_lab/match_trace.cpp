#include "match_trace.hpp"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct State {
    std::ofstream record;
    std::ifstream replay;
    std::string prefix;
    std::mutex mutex;
    std::uint64_t calls = 0;
    State() {
        if (const char* p = std::getenv("NSS_MATCH_RECORD")) { prefix = p; record.open(p, std::ios::binary); if (!record) throw std::runtime_error("trace open failed"); }
        if (const char* p = std::getenv("NSS_MATCH_REPLAY")) { if (prefix.empty()) prefix = p + std::string(".replay"); replay.open(p, std::ios::binary); if (!replay) throw std::runtime_error("replay open failed"); }
    }
    ~State() {
        if (prefix.empty()) return;
        std::ofstream meta(prefix + ".meta.json");
        meta << "{\"calls\":" << calls << ",\"replay_consumed\":"
             << (!replay.is_open() || replay.peek() == std::char_traits<char>::eof() ? "true" : "false") << "}";
    }
};
State& state() { static State value; return value; }

void transfer(std::array<std::int32_t, 9>& header, std::uint64_t hash, std::vector<std::uint32_t>& payload) {
    auto& s = state();
    if (!s.record.is_open() && !s.replay.is_open()) return;
    std::lock_guard lock(s.mutex);
    ++s.calls;
    if (s.record.is_open()) {
        s.record.write(reinterpret_cast<const char*>(header.data()), sizeof(header));
        s.record.write(reinterpret_cast<const char*>(&hash), sizeof(hash));
        s.record.write(reinterpret_cast<const char*>(payload.data()), payload.size() * 4);
        if (!s.record) throw std::runtime_error("trace write failed");
    }
    if (s.replay.is_open()) {
        std::array<std::int32_t, 9> previous;
        std::uint64_t previous_hash;
        s.replay.read(reinterpret_cast<char*>(previous.data()), sizeof(previous));
        s.replay.read(reinterpret_cast<char*>(&previous_hash), sizeof(previous_hash));
        if (!s.replay || !std::equal(header.begin(), header.begin() + 7, previous.begin()) ||
            previous[7] < 0 || previous[7] > header[4] || previous[8] < 0 || previous[8] > 1024 * 1024)
            throw std::runtime_error("replay geometry/count mismatch at call " + std::to_string(s.calls));
        header = previous;
        payload.resize(header[8]);
        s.replay.read(reinterpret_cast<char*>(payload.data()), payload.size() * 4);
        if (!s.replay) throw std::runtime_error("replay truncated");
    }
}
}

std::uint64_t nss_trace_hash(const float* p, int width, int height, int stride) {
    // Full 1080p iterative replay may disable repeated whole-frame hashing.
    // Such traces prove coordinates/replay consumption, not per-query input
    // identity; the worker separately records the exact source payload hash.
    static const bool omit = std::getenv("NSS_MATCH_OMIT_INPUT_HASH") != nullptr;
    if (omit) return 0;
    std::uint64_t hash = 14695981039346656037ull;
    if (!p || width < 0 || height < 0 || stride < width) return 0;
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
        std::uint32_t bits; std::memcpy(&bits, p + y * stride + x, 4);
        hash ^= bits; hash *= 1099511628211ull;
    }
    return hash;
}

int nss_trace_matches(int kind, int x, int y, int block, int capacity, int width, int height,
                       std::uint64_t input_hash, nss::Match* matches, int count) {
    if (count < 0 || count > capacity || capacity > 64) throw std::runtime_error("invalid matcher count");
    std::array<std::int32_t, 9> header{kind, x, y, block, capacity, width, height, count, 5 * count};
    std::vector<std::uint32_t> payload(5 * count);
    for (int i = 0; i < count; ++i) {
        payload[5*i] = matches[i].x; payload[5*i+1] = matches[i].y; payload[5*i+2] = matches[i].t;
        payload[5*i+3] = matches[i].ordinal; std::memcpy(&payload[5*i+4], &matches[i].dist, 4);
    }
    transfer(header, input_hash, payload);
    if (header[8] != 5 * header[7]) throw std::runtime_error("invalid matcher payload");
    for (int i = 0; i < header[7]; ++i) {
        matches[i].x = int(payload[5*i]); matches[i].y = int(payload[5*i+1]); matches[i].t = int(payload[5*i+2]);
        matches[i].ordinal = payload[5*i+3]; std::memcpy(&matches[i].dist, &payload[5*i+4], 4);
    }
    return header[7];
}

void nss_trace_indices(int m, int n, int q, std::uint64_t input_hash, int* indices) {
    std::array<std::int32_t, 9> header{6, m, n, q, m*q, 0, 0, m*q, m*q};
    std::vector<std::uint32_t> payload(indices, indices + m*q);
    transfer(header, input_hash, payload);
    if (header[8] != m*q || header[7] != m*q) throw std::runtime_error("invalid pixel matcher payload");
    for (int i = 0; i < m*q; ++i) indices[i] = int(payload[i]);
}
