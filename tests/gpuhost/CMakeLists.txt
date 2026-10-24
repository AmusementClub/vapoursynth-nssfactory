# SPDX-License-Identifier: GPL-2.0-only
# Backend-free host helpers (src/gpuhost, D27); no device needed.
add_executable(test_gpuhost test_chunk_cache.cpp)
target_link_libraries(test_gpuhost PRIVATE nss_gpuhost Threads::Threads)
add_test(NAME test_gpuhost COMMAND test_gpuhost)
set_tests_properties(test_gpuhost PROPERTIES LABELS "cuda;vk")
