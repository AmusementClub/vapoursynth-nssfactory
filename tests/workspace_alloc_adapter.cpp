// Substitute only the allocator, using the production Workspace implementation.
// This supports deterministic ENOMEM on both Mach-O and ELF linkers.
#include <cstdlib>
int workspace_test_memalign(void**, std::size_t, std::size_t);
#define posix_memalign workspace_test_memalign
#include "../src/host/workspace.cpp"
#undef posix_memalign
