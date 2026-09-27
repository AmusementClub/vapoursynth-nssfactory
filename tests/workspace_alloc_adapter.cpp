// SPDX-License-Identifier: GPL-2.0-only
// Substitute only the allocator, using the production Workspace implementation.
// This supports deterministic ENOMEM on both Mach-O and ELF linkers.
#include <cstdlib>
int workspace_test_memalign(void**, std::size_t, std::size_t);
#define NSS_WORKSPACE_ALLOC_ADAPTER 1
#define posix_memalign workspace_test_memalign
#include "../src/host/workspace.cpp"
#undef posix_memalign
