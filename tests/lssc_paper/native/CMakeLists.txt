add_library(nss_lssc_explore SHARED core.cpp learning.cpp api.cpp)
target_link_libraries(nss_lssc_explore PRIVATE nss_cpu)
target_include_directories(nss_lssc_explore PRIVATE "${PROJECT_SOURCE_DIR}/src")
# Rank/feasibility guards and FP64 accumulators must retain IEEE semantics.
target_compile_options(nss_lssc_explore PRIVATE -O3 -fno-fast-math -ffp-contract=off)
set_target_properties(nss_lssc_explore PROPERTIES CXX_VISIBILITY_PRESET hidden
                      VISIBILITY_INLINES_HIDDEN ON)

find_package(Threads REQUIRED)
add_executable(test_lssc_native_smoke smoke.cpp)
target_link_libraries(test_lssc_native_smoke PRIVATE nss_lssc_explore Threads::Threads)
add_test(NAME test_lssc_native_smoke COMMAND test_lssc_native_smoke)
