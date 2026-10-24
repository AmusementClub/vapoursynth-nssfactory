// SPDX-License-Identifier: GPL-2.0-only
// SlotPool hands out the slot free the longest: leases taken and given back
// one after another rotate over every slot (a LIFO pool returned the same
// slot each time, and so one CUDA stream to every frame). No device needed.
#include "cuda/runtime/stream_pool.hpp"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <set>
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
    if (failures) {
        std::printf("FAIL: lease order");
        for (int id : order) std::printf(" %d", id);
        std::printf("\n");
        return EXIT_FAILURE;
    }
    std::printf("cuda slot pool OK\n");
    return EXIT_SUCCESS;
}
