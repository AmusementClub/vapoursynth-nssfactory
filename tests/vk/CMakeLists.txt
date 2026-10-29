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
# The other queue layouts (D28): one copy queue (downloads deferred until the
# kernel is done) with staged uploads, and everything on the compute queue.
add_test(NAME test_vk_runtime_one_copy_queue COMMAND test_vk_runtime)
add_test(NAME test_vk_runtime_compute_queue COMMAND test_vk_runtime)
set_tests_properties(test_vk_runtime_one_copy_queue PROPERTIES LABELS vk SKIP_RETURN_CODE 77
                     ENVIRONMENT "NSS_VK_COPY_QUEUES=1;NSS_VK_REBAR=0")
set_tests_properties(test_vk_runtime_compute_queue PROPERTIES LABELS vk SKIP_RETURN_CODE 77
                     ENVIRONMENT "NSS_VK_COPY_QUEUES=0")

# Frames retained by a slot (direct I/O) across completed, unsubmitted and
# failed waits; CPU (lavapipe) device only, skip 77 otherwise.
add_executable(test_vk_slot_frames test_vk_slot_frames.cpp)
target_link_libraries(test_vk_slot_frames PRIVATE nss_vk_core)
add_test(NAME test_vk_slot_frames COMMAND test_vk_slot_frames)
set_tests_properties(test_vk_slot_frames PROPERTIES LABELS vk SKIP_RETURN_CODE 77)

# PNLM's arithmetic (shaders/pnlm/common.glsl) against the model, bit for bit (V15).
add_executable(test_vk_pnlm_math test_vk_pnlm_math.cpp)
target_link_libraries(test_vk_pnlm_math PRIVATE nss_vk_core)
target_include_directories(test_vk_pnlm_math PRIVATE "${PROJECT_SOURCE_DIR}/include")
add_test(NAME test_vk_pnlm_math COMMAND test_vk_pnlm_math)
set_tests_properties(test_vk_pnlm_math PROPERTIES LABELS vk SKIP_RETURN_CODE 77)

# D23 warp layer, native against emulated against the host (V3).
add_executable(test_vk_warp test_vk_warp.cpp)
target_link_libraries(test_vk_warp PRIVATE nss_vk_core)
add_test(NAME test_vk_warp COMMAND test_vk_warp)
set_tests_properties(test_vk_warp PROPERTIES LABELS vk SKIP_RETURN_CODE 77)

# Lane matchers against the CPU matcher (V3; needs the CPU tree).
if(TARGET nss_cpu)
  add_executable(test_vk_match test_vk_match.cpp)
  target_link_libraries(test_vk_match PRIVATE nss_vk_core nss_cpu)
  target_include_directories(test_vk_match PRIVATE "${PROJECT_SOURCE_DIR}/include" "${PROJECT_SOURCE_DIR}/src")
  add_test(NAME test_vk_match COMMAND test_vk_match)
  set_tests_properties(test_vk_match PROPERTIES LABELS vk SKIP_RETURN_CODE 77)
endif()

# DCT lines and fixed-point aggregation against the CPU and host models (V3;
# needs the CPU tree).
if(TARGET nss_cpu)
  add_executable(test_vk_aggregate test_vk_aggregate.cpp)
  target_link_libraries(test_vk_aggregate PRIVATE nss_vk_core nss_cpu)
  target_include_directories(test_vk_aggregate PRIVATE "${PROJECT_SOURCE_DIR}/include" "${PROJECT_SOURCE_DIR}/src")
  add_test(NAME test_vk_aggregate COMMAND test_vk_aggregate)
  set_tests_properties(test_vk_aggregate PROPERTIES LABELS vk SKIP_RETURN_CODE 77)
endif()

# BM3D group kernels against the CPU's group filter (V5; needs the CPU tree).
if(TARGET nss_cpu)
  add_executable(test_vk_bm3d test_vk_bm3d.cpp)
  target_link_libraries(test_vk_bm3d PRIVATE nss_vk_core nss_cpu)
  target_include_directories(test_vk_bm3d PRIVATE "${PROJECT_SOURCE_DIR}/include" "${PROJECT_SOURCE_DIR}/src")
  add_test(NAME test_vk_bm3d COMMAND test_vk_bm3d)
  set_tests_properties(test_vk_bm3d PROPERTIES LABELS vk SKIP_RETURN_CODE 77)
endif()

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
set(NSS_VK_PENDING_FILTERS "WNNM,MCWNNM,NCSR,NLH,TWSC,LSSC,DA3D,BM4D")
add_test(NAME test_vk_plugin_interface
         COMMAND ${Python3_EXECUTABLE} "${PROJECT_SOURCE_DIR}/tests/test_plugin_interface.py"
                 --plugin "${NSS_VK_PLUGIN}" --namespace nss_vk
                 --backend-args device_id,num_streams --expect-missing "${NSS_VK_PENDING_FILTERS}"
                 --cpu-only VAggregate
                 --default-temporal rolling --rolling-cache adaptive)
set_tests_properties(test_vk_backend test_vk_plugin_interface PROPERTIES LABELS vk SKIP_RETURN_CODE 77)
# VapourSynth R80+ GPU frames (D29). With the configured Python's core older
# than R80 (the dev host's R78 venv) it checks that the same binary registers
# plain `vnode` arguments; NSS_VK_GPU_FRAMES_PYTHON names a Python whose
# VapourSynth is R80 or later (dev host: ~/src/vs-envs/r81/python) for the
# GPU-frame cases.
set(NSS_VK_GPU_FRAMES_PYTHON "" CACHE STRING "Python with VapourSynth R80+ for test_vk_gpu_frames_r8x (optional)")
set(_nss_vk_gpu_args --vk "${NSS_VK_PLUGIN}")
if(TARGET nss)
  list(APPEND _nss_vk_gpu_args --cpu "$<TARGET_FILE:nss>")
endif()
add_test(NAME test_vk_gpu_frames
         COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_vk_gpu_frames.py" ${_nss_vk_gpu_args})
set_tests_properties(test_vk_gpu_frames PROPERTIES LABELS vk SKIP_RETURN_CODE 77)

add_test(NAME test_vk_stress_harness
         COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_vk_stress_harness.py")
set_tests_properties(test_vk_stress_harness PROPERTIES LABELS vk TIMEOUT 30)

# D30 r3 preview: repeatedly create/free and render NLM, BM3D and PNLM.  The
# test runs on a real Vulkan device (including GPU frames where supported) and
# skips cleanly in build-only CI.
add_test(NAME test_vk_stress
         COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_vk_stress.py"
                 --plugin "${NSS_VK_PLUGIN}")
set_tests_properties(test_vk_stress PROPERTIES LABELS vk SKIP_RETURN_CODE 77 TIMEOUT 600)
if(NSS_VK_GPU_FRAMES_PYTHON)
  add_test(NAME test_vk_gpu_frames_r8x
           COMMAND ${NSS_VK_GPU_FRAMES_PYTHON} "${CMAKE_CURRENT_SOURCE_DIR}/test_vk_gpu_frames.py" ${_nss_vk_gpu_args})
  set_tests_properties(test_vk_gpu_frames_r8x PROPERTIES LABELS vk SKIP_RETURN_CODE 77)
  add_test(NAME test_vk_stress_r8x
           COMMAND ${NSS_VK_GPU_FRAMES_PYTHON} "${CMAKE_CURRENT_SOURCE_DIR}/test_vk_stress.py"
                   --plugin "${NSS_VK_PLUGIN}" --require-gpu)
  set_tests_properties(test_vk_stress_r8x PROPERTIES LABELS vk TIMEOUT 600)
endif()
# Filters against the CPU plugin (needs the CPU tree).
if(TARGET nss)
  foreach(filter nlm bm3d pnlm)
    add_test(NAME test_vk_${filter}_live
             COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_vk_${filter}_live.py"
                     --cpu "$<TARGET_FILE:nss>" --vk "${NSS_VK_PLUGIN}")
    set_tests_properties(test_vk_${filter}_live PROPERTIES LABELS vk SKIP_RETURN_CODE 77)
    # Staged uploads on one copy queue: the cross-slot upload waits and the
    # deferred downloads.
    add_test(NAME test_vk_${filter}_live_staged
             COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_vk_${filter}_live.py"
                     --cpu "$<TARGET_FILE:nss>" --vk "${NSS_VK_PLUGIN}" --quick)
    set_tests_properties(test_vk_${filter}_live_staged PROPERTIES LABELS vk SKIP_RETURN_CODE 77
                         ENVIRONMENT "NSS_VK_REBAR=0;NSS_VK_COPY_QUEUES=1")
  endforeach()
  # PNLM's emulated warp-layer kernels on a native device.
  add_test(NAME test_vk_pnlm_live_emulated
           COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_vk_pnlm_live.py"
                   --cpu "$<TARGET_FILE:nss>" --vk "${NSS_VK_PLUGIN}" --quick)
  set_tests_properties(test_vk_pnlm_live_emulated PROPERTIES LABELS vk SKIP_RETURN_CODE 77
                       ENVIRONMENT "NSS_VK_FORCE_EMULATED_WARP=1")
endif()
