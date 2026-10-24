# SPDX-License-Identifier: GPL-2.0-only
# Vulkan backend tests (label "vk"). Tests that need a device exit 77 (skip)
# without one; GitHub CI only builds nss_vk and runs the device-free tests
# (owner, 2026-10-24), the dev host runs the rest.

add_executable(test_vk_support test_vk_support.cpp)
target_link_libraries(test_vk_support PRIVATE nss_vk_core)
add_test(NAME test_vk_support COMMAND test_vk_support)
set_tests_properties(test_vk_support PROPERTIES LABELS vk)

# Needs a supported device (skip 77 otherwise): budget, transfers, slots,
# frames in flight, create/free, pipeline cache, timestamps.
add_executable(test_vk_runtime test_vk_runtime.cpp)
target_link_libraries(test_vk_runtime PRIVATE nss_vk_core)
add_test(NAME test_vk_runtime COMMAND test_vk_runtime)
set_tests_properties(test_vk_runtime PROPERTIES LABELS vk SKIP_RETURN_CODE 77)

# Queue-layout throughput (D28), run by hand on a device; not a test.
add_executable(bench_vk_queues bench_vk_queues.cpp)
target_link_libraries(bench_vk_queues PRIVATE nss_vk_core)

find_package(Python3 COMPONENTS Interpreter QUIET)
if(NOT Python3_Interpreter_FOUND)
  message(WARNING "Python3 interpreter unavailable: Vulkan plugin tests are not registered")
  return()
endif()

set(NSS_VK_PLUGIN "$<TARGET_FILE:nss_vk>")
add_test(NAME test_vk_backend
         COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_vk_backend.py" --plugin "${NSS_VK_PLUGIN}")
# D21 parity against the CPU golden. Filters still pending in
# docs/vulkan-plan.md are expected to be missing; each phase removes the
# filters it delivers. VAggregate is the CPU's alone (D21).
set(NSS_VK_PENDING_FILTERS "BM3D,NLM,WNNM,MCWNNM,NCSR,NLH,TWSC,LSSC,DA3D,BM4D")
add_test(NAME test_vk_plugin_interface
         COMMAND ${Python3_EXECUTABLE} "${PROJECT_SOURCE_DIR}/tests/test_plugin_interface.py"
                 --plugin "${NSS_VK_PLUGIN}" --namespace nss_vk
                 --backend-args device_id,num_streams --expect-missing "${NSS_VK_PENDING_FILTERS}"
                 --cpu-only VAggregate
                 --default-temporal rolling --rolling-cache adaptive)
set_tests_properties(test_vk_backend test_vk_plugin_interface PROPERTIES LABELS vk SKIP_RETURN_CODE 77)
