// SPDX-License-Identifier: GPL-2.0-only
// SlotPool hands out the slot free the longest: leases taken and given back
// one after another rotate over every slot (a LIFO pool returned the same
// slot each time, and so one CUDA stream to every frame). acquire(order)
// serves the waiting threads lowest order first. No device needed.
#include "cuda/runtime/stream_pool.hpp"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <chrono>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

namespace {

struct Slot {
    int id = 0;
};

}  // namespace

int main() {
    std::vector<std::unique_ptr<Slot>> slots;
    for (int i = 0; i < 3; ++i) slots.push_back(std::make_unique<Slot>(Slot{i}));
    nss_cuda::SlotPool<Slot> pool(std::move(slots));
    std::vector<int> order;
    for (int frame = 0; frame < 9; ++frame) {
        auto lease = pool.acquire();  // given back at the end of the iteration
        order.push_back(lease->id);
    }
    int failures = 0;
    for (std::size_t i = 3; i < order.size(); ++i) {
        if (order[i] != order[i - 3]) ++failures;  // period 3: every slot in turn
    }
    if (std::set<int>(order.begin(), order.end()).size() != 3) ++failures;
    // Held leases still get distinct slots.
    {
        auto a = pool.acquire(), b = pool.acquire(), c = pool.acquire();
        if (a->id == b->id || b->id == c->id || a->id == c->id) ++failures;
    }
    // Ordered: with the one slot held, threads wait with orders 5, 2, 8, 1;
    // given back, the slot goes to them as 1, 2, 5, 8.
    std::vector<std::unique_ptr<Slot>> one;
    one.push_back(std::make_unique<Slot>(Slot{0}));
    nss_cuda::SlotPool<Slot> ordered(std::move(one));
    std::vector<int> served;
    std::mutex served_mu;
    {
        auto held = ordered.acquire(0);
        std::vector<std::thread> threads;
        for (int order : {5, 2, 8, 1}) {
            threads.emplace_back([&, order] {
                auto lease = ordered.acquire(order);
                std::lock_guard lock(served_mu);
                served.push_back(order);
            });
            std::this_thread::sleep_for(std::chrono::milliseconds(20));  // all four wait before the slot is free
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        { auto release = std::move(held); }
        for (auto& t : threads) t.join();
    }
    if (served != std::vector<int>{1, 2, 5, 8}) {
        std::printf("FAIL: ordered acquire served");
        for (int o : served) std::printf(" %d", o);
        std::printf("\n");
        ++failures;
    }
    if (failures) {
        std::printf("FAIL: lease order");
        for (int id : order) std::printf(" %d", id);
        std::printf("\n");
        return EXIT_FAILURE;
    }
    std::printf("cuda slot pool OK\n");
    return EXIT_SUCCESS;
}
