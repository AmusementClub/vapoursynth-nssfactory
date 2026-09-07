// Linux validation-only allocator interposer. Only calls originating in NSS_SO
// can fail; VapourSynth/Python allocations continue to use libc normally.
#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>

namespace {
std::atomic<long> remaining{-1};
std::atomic<long> injected{0};
}
extern "C" void nss_test_arm_workspace_failure(long after) noexcept {
    injected.store(0);
    remaining.store(after);
}
extern "C" long nss_test_workspace_injections() noexcept { return injected.load(); }
extern "C" int posix_memalign(void** pointer, std::size_t alignment, std::size_t size) noexcept {
    using Function = int (*)(void**, std::size_t, std::size_t);
    static auto real = reinterpret_cast<Function>(dlsym(RTLD_NEXT, "posix_memalign"));
    const char* plugin = std::getenv("NSS_SO");
    if (plugin && remaining.load() >= 0) {
        Dl_info caller{};
        if (dladdr(__builtin_return_address(0), &caller) && caller.dli_fname &&
            std::strcmp(caller.dli_fname, plugin) == 0 && remaining.fetch_sub(1) == 0) {
            injected.fetch_add(1);
            return ENOMEM;
        }
    }
    return real ? real(pointer, alignment, size) : ENOMEM;
}
