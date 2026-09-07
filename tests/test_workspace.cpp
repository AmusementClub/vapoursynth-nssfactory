#include "nss/workspace.hpp"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <new>

namespace {
bool fail_allocation = false;
int allocations = 0;
bool check(bool condition, const char* message) {
    if (!condition) std::fprintf(stderr, "%s\n", message);
    return condition;
}
}

int workspace_test_memalign(void** p, std::size_t alignment, std::size_t bytes) {
    ++allocations;
    if (fail_allocation) return ENOMEM;
    return posix_memalign(p, alignment, bytes);
}

int main() {
    bool ok = true;
    nss::Workspace workspace;
    float* original = workspace.get(17);
    original[16] = 42.f;
    const int before = allocations;
    const std::size_t limit = std::numeric_limits<std::size_t>::max() / sizeof(float);
    for (std::size_t size : {limit + 1, std::numeric_limits<std::size_t>::max()}) {
        bool rejected = false;
        try { (void)workspace.get(size); } catch (const std::bad_alloc&) { rejected = true; }
        ok = check(rejected, "overflowing float count was accepted") && ok;
    }
    ok = check(allocations == before, "overflow reached allocator") && ok;
    float* retained = workspace.get(17);
    if (retained == original && allocations == before) {
        ok = check(retained[16] == 42.f, "overflow changed existing contents") && ok;
    } else {
        ok = check(false, "overflow replaced existing buffer") && ok;
    }
    fail_allocation = true;
    bool rejected = false;
    try { (void)workspace.get(1024); } catch (const std::bad_alloc&) { rejected = true; }
    ok = check(rejected, "allocation failure was accepted") && ok;
    fail_allocation = false;
    const int after_failure = allocations;
    retained = workspace.get(17);
    if (retained == original && allocations == after_failure) {
        ok = check(retained[16] == 42.f, "failed growth changed existing contents") && ok;
    } else {
        ok = check(false, "failed growth discarded existing buffer") && ok;
    }
    float* grown = workspace.get(1024);
    grown[1023] = 7.f;
    ok = check(workspace.get(1024) == grown && grown[1023] == 7.f, "growth retry failed") && ok;
    ok = check(reinterpret_cast<std::uintptr_t>(grown) % 64 == 0, "alignment lost") && ok;
    nss::Workspace empty;
    float* zero = empty.get(0);
    ok = check(zero != nullptr && empty.get(0) == zero, "zero request is not stable") && ok;
    nss::Workspace failed_first;
    fail_allocation = true;
    rejected = false;
    try { (void)failed_first.get(17); } catch (const std::bad_alloc&) { rejected = true; }
    fail_allocation = false;
    ok = check(rejected, "initial allocation failure was accepted") && ok;
    ok = check(failed_first.get(0) != nullptr, "failed first allocation left a null reusable buffer") && ok;
    return ok ? 0 : 1;
}
