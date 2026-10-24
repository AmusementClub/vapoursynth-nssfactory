// SPDX-License-Identifier: GPL-2.0-only
// gpuhost chunk cache (D27) on a plain host block type: LRU order, growth
// after repeated misses on dropped chunks, block reuse through the pool,
// one computation per chunk under concurrent requests, raster grid.
#include "gpuhost/chunk_cache.hpp"
#include "gpuhost/geometry.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

namespace {

int failures = 0;
void expect(bool condition, const char* what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

// A host block: get() null when empty, as<T>() its bytes (as PinnedBuffer).
struct Block {
    std::vector<std::uint8_t> data;
    Block() = default;
    explicit Block(std::size_t bytes) : data(bytes) {}
    void* get() const noexcept { return data.empty() ? nullptr : const_cast<std::uint8_t*>(data.data()); }
    template <class T> T* as() const noexcept { return static_cast<T*>(get()); }
};

}  // namespace

int main() {
    using Cache = nss::gpuhost::ChunkCache<Block>;
    std::atomic<int> allocations{0};
    Cache cache;
    cache.configure(4, 16, {0, 4, 8}, 2, 3, [&](std::size_t bytes) {
        ++allocations;
        return Block(bytes);
    });
    std::atomic<int> computed{0};
    auto chunk = [&](int start) {
        return cache.get(start, [&] {
            ++computed;
            auto store = cache.make_store(start, 4);
            store->plane(0, 1)[0] = static_cast<float>(start);
            return store;
        });
    };
    expect(chunk(0)->plane(0, 1)[0] == 0.0f, "store holds what compute wrote");
    chunk(4);
    chunk(0);   // hit: 0 becomes most recent
    chunk(8);   // drops 4 (least recent)
    expect(computed == 3, "hits do not recompute");
    chunk(0);
    expect(computed == 3, "0 stayed cached");
    chunk(4);   // first miss on a dropped chunk: recomputed, drops 8, no growth yet
    chunk(8);   // second miss on a dropped chunk: the cache grows to 3 (nothing dropped)
    chunk(0);   // hit
    const int before = computed;
    chunk(4);
    chunk(8);
    chunk(0);
    expect(computed == before, "grown to three chunks: all three cached");
    expect(allocations < computed, "dropped chunks give their blocks back to the pool");

    // Concurrent requests for one chunk compute it once.
    Cache shared;
    shared.configure(2, 8, {0, 0, 0}, 4, 4, [](std::size_t bytes) { return Block(bytes); });
    std::atomic<int> once{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t) {
        threads.emplace_back([&] {
            shared.get(10, [&] {
                ++once;
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                return shared.make_store(10, 2);
            });
        });
    }
    for (std::thread& thread : threads) thread.join();
    expect(once == 1, "one computation under concurrent requests");

    // Raster grid: last row/column clamped to (width - block, height - block).
    const auto grid = nss::gpuhost::make_raster_grid(21, 10, 8, 4);
    expect(grid.nx == 5 && grid.ny == 2 && grid.x(4) == 13 && grid.y(5) == 2 && grid.count() == 10, "raster grid");
    expect(nss::gpuhost::make_raster_grid(5, 10, 8, 4).count() == 0, "plane smaller than a block");

    if (failures) return EXIT_FAILURE;
    std::printf("gpuhost OK\n");
    return EXIT_SUCCESS;
}
