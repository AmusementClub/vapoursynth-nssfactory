// SPDX-License-Identifier: GPL-2.0-only
// Runtime layer: frame I/O round trip through pinned staging with odd host
// strides and pitched device memory, budget charging/release, arena bounds,
// and SlotPool exclusivity under concurrent leases.
#include "cuda/runtime/device.hpp"
#include "cuda/runtime/frame_io.hpp"
#include "cuda/runtime/memory.hpp"
#include "cuda/runtime/stream_pool.hpp"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

int failures = 0;
#define CHECK(cond, msg)                      \
    do {                                      \
        if (!(cond)) {                        \
            std::printf("FAIL: %s\n", msg);   \
            ++failures;                       \
        }                                     \
    } while (0)

void frame_io_round_trip() {
    const int width = 77, height = 31, host_stride = 96;  // floats; stride != width
    std::vector<float> host(static_cast<std::size_t>(host_stride) * height, -1.f), back(host.size(), -2.f);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) host[static_cast<std::size_t>(y) * host_stride + x] = y * 1000.f + x;
    void* device = nullptr;
    std::size_t pitch = 0;
    NSS_CUDA_CHECK(cudaMallocPitch(&device, &pitch, width * sizeof(float), height));
    nss_cuda::PinnedBuffer staging_in(width * sizeof(float) * height), staging_out(width * sizeof(float) * height);
    nss_cuda::Stream stream;
    nss_cuda::upload_plane(host.data(), host_stride * sizeof(float), width * sizeof(float), height, staging_in.get(),
                           device, pitch, stream);
    nss_cuda::begin_download(device, pitch, width * sizeof(float), height, staging_out.get(), stream);
    stream.synchronize();
    nss_cuda::finish_download(staging_out.get(), width * sizeof(float), height, back.data(), host_stride * sizeof(float));
    bool same = true;
    for (int y = 0; y < height; ++y) {
        same &= std::memcmp(&host[static_cast<std::size_t>(y) * host_stride], &back[static_cast<std::size_t>(y) * host_stride],
                            width * sizeof(float)) == 0;
        same &= back[static_cast<std::size_t>(y) * host_stride + width] == -2.f;  // padding untouched
    }
    CHECK(same, "frame_io round trip altered pixels or wrote past the row");
    cudaFree(device);
}

void budget_and_arena() {
    auto budget = std::make_shared<nss::ResourceBudget>(1u << 20);
    {
        nss_cuda::DeviceBuffer a(600 * 1024, budget);
        CHECK(budget->snapshot().owned == 600 * 1024, "device buffer not charged");
        bool threw = false;
        try {
            nss_cuda::DeviceBuffer b(600 * 1024, budget);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        CHECK(threw, "budget overrun not rejected");
        CHECK(budget->snapshot().owned == 600 * 1024, "failed allocation leaked a charge");
        nss_cuda::PinnedBuffer pinned(100 * 1024, budget);
        CHECK(budget->snapshot().bytes[static_cast<std::size_t>(nss::ResourceKind::Pinned)] == 100 * 1024,
              "pinned buffer not charged as Pinned");
    }
    CHECK(budget->snapshot().owned == 0, "buffers not released from the budget");

    nss_cuda::DeviceArena arena(1000, nullptr);
    CHECK(arena.capacity() == 1024, "arena capacity not rounded to 256");
    arena.take<float>(100);
    CHECK(arena.used() == 512, "arena allocation not aligned");
    bool threw = false;
    try {
        arena.take<float>(200);
    } catch (const std::logic_error&) {
        threw = true;
    }
    CHECK(threw, "arena overflow not detected");
}

void slot_pool() {
    struct Slot {
        std::atomic<int> users{0};
    };
    std::vector<std::unique_ptr<Slot>> slots;
    for (int i = 0; i < 3; ++i) slots.push_back(std::make_unique<Slot>());
    nss_cuda::SlotPool<Slot> pool(std::move(slots));
    std::atomic<int> concurrent{0}, peak{0}, violations{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < 12; ++t) {
        threads.emplace_back([&] {
            for (int i = 0; i < 200; ++i) {
                auto lease = pool.acquire();
                if (lease->users.fetch_add(1) != 0) violations++;
                const int now = ++concurrent;
                int old = peak.load();
                while (now > old && !peak.compare_exchange_weak(old, now)) {}
                std::this_thread::yield();
                --concurrent;
                lease->users.fetch_sub(1);
            }
        });
    }
    for (auto& thread : threads) thread.join();
    CHECK(violations == 0, "a slot was leased twice");
    CHECK(peak <= 3, "more concurrent leases than slots");
}

}  // namespace

int main() {
    if (nss_cuda::runtime_info().device_count == 0) {
        std::printf("test_cuda_runtime SKIP: no CUDA device\n");
        return 77;
    }
    frame_io_round_trip();
    budget_and_arena();
    slot_pool();
    std::printf("test_cuda_runtime: %d failures\n", failures);
    return failures ? 1 : 0;
}
