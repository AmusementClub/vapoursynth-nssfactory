# SPDX-License-Identifier: GPL-2.0-only
# CUDA backend tests (label "cuda"). Every test exits 77 (skip) without a CUDA
# device, so CUDA-enabled builds on GPU-less machines still pass ctest.

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
set(NSS_CUDA_PENDING_FILTERS "BM3D,VAggregate,NLM,WNNM,MCWNNM,NCSR,TWSC,NLH,LSSC")
add_test(NAME test_cuda_plugin_interface
         COMMAND ${Python3_EXECUTABLE} "${PROJECT_SOURCE_DIR}/tests/test_plugin_interface.py"
                 --plugin "${NSS_CUDA_PLUGIN}" --namespace nss_cuda
                 --backend-args device_id,num_streams --expect-missing "${NSS_CUDA_PENDING_FILTERS}")
set_tests_properties(test_cuda_backend test_cuda_backend_forced_jit test_cuda_plugin_interface PROPERTIES
  LABELS cuda SKIP_RETURN_CODE 77)
