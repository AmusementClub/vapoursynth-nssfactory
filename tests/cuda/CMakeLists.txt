# SPDX-License-Identifier: GPL-2.0-only
# CUDA backend tests (label "cuda"). Every test exits 77 (skip) without a CUDA
# device, so CUDA-enabled builds on GPU-less machines still pass ctest.

# Primitive tests compare against the CPU kernels, so they need the CPU tree.
if(TARGET nss_cpu)
  add_executable(test_cuda_match test_cuda_match.cpp)
  target_link_libraries(test_cuda_match PRIVATE nss_cuda_core nss_cpu)
  target_include_directories(test_cuda_match PRIVATE "${PROJECT_SOURCE_DIR}/include" "${PROJECT_SOURCE_DIR}/src")
  add_test(NAME test_cuda_match COMMAND test_cuda_match)
  set_tests_properties(test_cuda_match PROPERTIES LABELS cuda SKIP_RETURN_CODE 77)
endif()

foreach(test test_cuda_aggregate test_cuda_runtime)
  add_executable(${test} ${test}.cpp)
  target_link_libraries(${test} PRIVATE nss_cuda_core)
  add_test(NAME ${test} COMMAND ${test})
  set_tests_properties(${test} PROPERTIES LABELS cuda SKIP_RETURN_CODE 77)
endforeach()

find_package(Python3 COMPONENTS Interpreter QUIET)
if(NOT Python3_Interpreter_FOUND)
  message(WARNING "Python3 interpreter unavailable: CUDA plugin tests are not registered")
  return()
endif()

set(NSS_CUDA_PLUGIN "$<TARGET_FILE:nss_cuda>")
add_test(NAME test_cuda_backend
         COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_cuda_backend.py" --plugin "${NSS_CUDA_PLUGIN}")
add_test(NAME test_cuda_backend_forced_jit
         COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_cuda_backend.py" --plugin "${NSS_CUDA_PLUGIN}" --force-jit)
set_tests_properties(test_cuda_backend_forced_jit PROPERTIES ENVIRONMENT "CUDA_FORCE_PTX_JIT=1")
# D14 parity against the CPU golden. Filters still pending in docs/cuda-plan.md
# are expected to be missing; each phase removes the filters it delivers.
set(NSS_CUDA_PENDING_FILTERS "LSSC")
add_test(NAME test_cuda_plugin_interface
         COMMAND ${Python3_EXECUTABLE} "${PROJECT_SOURCE_DIR}/tests/test_plugin_interface.py"
                 --plugin "${NSS_CUDA_PLUGIN}" --namespace nss_cuda
                 --backend-args device_id,num_streams --expect-missing "${NSS_CUDA_PENDING_FILTERS}")
set_tests_properties(test_cuda_backend test_cuda_backend_forced_jit test_cuda_plugin_interface PROPERTIES
  LABELS cuda SKIP_RETURN_CODE 77)
# Live CPU-vs-GPU filter checks need the CPU plugin as the reference.
if(TARGET nss)
  foreach(filter bm3d nlm wnnm mcwnnm ncsr nlh twsc)
    add_test(NAME test_cuda_${filter}_live
             COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_cuda_${filter}_live.py"
                     --cpu "$<TARGET_FILE:nss>" --cuda "${NSS_CUDA_PLUGIN}")
    set_tests_properties(test_cuda_${filter}_live PROPERTIES LABELS cuda SKIP_RETURN_CODE 77)
  endforeach()
endif()
