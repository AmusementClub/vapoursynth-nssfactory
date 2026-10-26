// SPDX-License-Identifier: GPL-2.0-only
// Slot::retain: the frames a frame's work reads or writes (direct I/O) are
// let go once the work is known done (a completed wait, or commands that were
// never submitted), and leaked after the slot's own wait failed, so that VapourSynth does
// not reuse memory the device may still copy from or into. A gate semaphore
// holds back valid work past a 200 ms wait bound; nothing faults. Runs on a CPU
// (lavapipe) device only: the failed wait poisons the device. Exits 77
// without one.
#include "vulkan/runtime/context.hpp"
#include "vulkan/runtime/instance.hpp"
#include "vulkan/runtime/memory.hpp"
#include "vulkan/runtime/slot.hpp"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace {

using namespace nss_vk;

int failures = 0;
int freed = 0;

void expect(bool condition, const std::string& what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

const VSFrame* VS_CC add_ref(const VSFrame* f) noexcept { return f; }
void VS_CC free_frame(const VSFrame*) noexcept { ++freed; }

void record_fill(Slot& slot, const Buffer& buffer, Slot::Stage stage) {
    slot.device().vk().vkCmdFillBuffer(slot.commands(stage), buffer.get(), 0, buffer.bytes(), 0x11111111u);
}

}  // namespace

int main() {
#ifdef _WIN32
    _putenv_s("NSS_VK_WAIT_TIMEOUT_MS", "200");
#else
    setenv("NSS_VK_WAIT_TIMEOUT_MS", "200", 1);
#endif
    const Instance& vk = instance();
    int index = -1;
    for (int i = 0; i < static_cast<int>(vk.devices.size()) && index < 0; ++i) {
        if (query_caps(vk.devices[static_cast<std::size_t>(i)]).type == VK_PHYSICAL_DEVICE_TYPE_CPU) index = i;
    }
    if (index < 0) {
        std::printf("vk slot frames SKIP: no CPU Vulkan device (the test poisons its device)\n");
        return 77;
    }
    std::shared_ptr<Device> device = Device::open(index, true);
    VSAPI api{};
    api.addFrameRef = add_ref;
    api.freeFrame = free_frame;
    const auto* frame = reinterpret_cast<const VSFrame*>(&api);  // any pointer: the fake API never reads it
    Buffer target(device, 4096, Memory::Device);

    {
        Slot slot(device);
        slot.retain(frame, &api);
        record_fill(slot, target, Slot::Upload);
        slot.submit();
        expect(freed == 0, "a retained frame stays while its work runs");
        slot.wait();
        expect(freed == 1, "a completed wait lets the frame go");
    }
    {
        Slot slot(device);
        slot.retain(frame, &api);
        record_fill(slot, target, Slot::Upload);
        slot.settle();  // as a lease returned by an exception before submit()
        expect(freed == 2, "commands that were never submitted let the frame go");
    }

    VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &type};
    VkSemaphore gate = VK_NULL_HANDLE;
    if (device->vk().vkCreateSemaphore(device->handle(), &info, nullptr, &gate) != VK_SUCCESS) return EXIT_FAILURE;
    std::pair<VkSemaphore, std::uint64_t> done{};
    {
        Slot slot(device);
        slot.retain(frame, &api);
        slot.wait_for(Slot::Upload, gate, 1);  // the work waits past the 200 ms bound
        record_fill(slot, target, Slot::Upload);
        slot.submit();
        done = slot.done(Slot::Upload);
        bool timed_out = false;
        try {
            slot.wait();
        } catch (const std::exception&) {
            timed_out = true;
        }
        expect(timed_out, "the wait for gated work fails");
        expect(device->poisoned(), "a failed wait poisons the device");
        slot.settle();  // as the lease during the unwind
    }
    expect(freed == 2, "after a failed wait the frame stays referenced, also once the slot is gone (freed " +
                           std::to_string(freed) + ")");
    // Let the gated work finish before the process ends.
    VkSemaphoreSignalInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
    signal.semaphore = gate;
    signal.value = 1;
    device->vk().vkSignalSemaphore(device->handle(), &signal);
    VkSemaphoreWaitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    wait.semaphoreCount = 1;
    wait.pSemaphores = &done.first;
    wait.pValues = &done.second;
    expect(device->vk().vkWaitSemaphores(device->handle(), &wait, 5'000'000'000ull) == VK_SUCCESS,
           "the gated work finishes once the gate opens");
    {
        // The device is poisoned now; a slot whose own wait sees its work
        // finish still lets its frame go (else every later frame would leak).
        Slot slot(device);
        slot.retain(frame, &api);
        record_fill(slot, target, Slot::Upload);
        slot.submit();
        slot.wait();
        expect(freed == 3, "on a poisoned device a completed wait still lets the frame go (freed " + std::to_string(freed) + ")");
    }
    if (failures) return EXIT_FAILURE;
    std::printf("vk slot frames OK\n");
    return EXIT_SUCCESS;
}
