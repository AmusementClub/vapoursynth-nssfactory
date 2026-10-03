# SPDX-License-Identifier: GPL-2.0-only
# CUDA backend libnss_cuda (namespace nss_cuda). Plan: docs/cuda-plan.md,
# decisions D13 (architectures/toolkit) and D18 (flag classes) in docs/plan.md.

include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
  message(FATAL_ERROR "NSS_ENABLE_CUDA=ON but no CUDA compiler was found; put nvcc on PATH or set CMAKE_CUDA_COMPILER")
endif()
enable_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
  message(FATAL_ERROR "libnss_cuda is built with nvcc only (found ${CMAKE_CUDA_COMPILER_ID})")
endif()
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.4)
  message(FATAL_ERROR "libnss_cuda requires CUDA 12.4 or newer (found ${CMAKE_CUDA_COMPILER_VERSION})")
endif()
find_package(CUDAToolkit REQUIRED)
# <windows.h> (pulled in by the CUDA headers) must not define min/max macros
# over std::min/std::max.
if(WIN32)
  add_compile_definitions(NOMINMAX)
endif()
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# D13: native cubins for the four supported generations plus compute_75 PTX as
# an untested best-effort JIT fallback for every other sm_75+ device.
# "default" applies that policy; any other value is passed through unchanged
# (e.g. -DNSS_CUDA_ARCHITECTURES=120-real for a faster single-GPU dev build).
set(NSS_CUDA_ARCHITECTURES "default" CACHE STRING
    "CUDA architectures for libnss_cuda; 'default' = 75/86/89/120 cubins + compute_75 PTX")
if(NSS_CUDA_ARCHITECTURES STREQUAL "default")
  set(NSS_CUDA_ARCH_LIST 75-real 86-real 89-real)
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
    list(APPEND NSS_CUDA_ARCH_LIST 120-real)
  else()
    message(WARNING "CUDA ${CMAKE_CUDA_COMPILER_VERSION} cannot generate sm_120; RTX 50-series GPUs will run the compute_75 PTX through the driver JIT")
  endif()
  list(APPEND NSS_CUDA_ARCH_LIST 75-virtual)
else()
  set(NSS_CUDA_ARCH_LIST ${NSS_CUDA_ARCHITECTURES})
endif()
# Split into cubin and PTX sets for the runtime support classification.
set(NSS_CUDA_CUBIN_ARCHS "")
set(NSS_CUDA_PTX_ARCHS "")
foreach(arch IN LISTS NSS_CUDA_ARCH_LIST)
  if(NOT arch MATCHES "^([0-9]+)(-real|-virtual)?$")
    message(FATAL_ERROR "Unsupported entry in NSS_CUDA_ARCHITECTURES: ${arch}")
  endif()
  set(number "${CMAKE_MATCH_1}")
  if(NOT CMAKE_MATCH_2 STREQUAL "-virtual")
    list(APPEND NSS_CUDA_CUBIN_ARCHS ${number})
  endif()
  if(NOT CMAKE_MATCH_2 STREQUAL "-real")
    list(APPEND NSS_CUDA_PTX_ARCHS ${number})
  endif()
endforeach()
string(JOIN "," NSS_CUDA_CUBIN_ARCHS_CSV ${NSS_CUDA_CUBIN_ARCHS})
string(JOIN "," NSS_CUDA_PTX_ARCHS_CSV ${NSS_CUDA_PTX_ARCHS})
string(JOIN "," NSS_CUDA_ARCH_STRING ${NSS_CUDA_ARCH_LIST})
message(STATUS "NSSFactory CUDA ${CMAKE_CUDA_COMPILER_VERSION}: architectures ${NSS_CUDA_ARCH_STRING}")

# Isolation guard (D3/D4): the CUDA tree reaches only the backend-neutral
# frontend and params headers, never CPU kernels or CPU host code.
file(GLOB_RECURSE NSS_CUDA_GUARDED CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*")
nss_forbid_cpu_includes("CUDA backend" ${NSS_CUDA_GUARDED})

# D18 flag classes. Every .cu belongs to exactly one class; configure fails on
# an unassigned or doubly-assigned file. CUDA_IEEE keeps IEEE division/sqrt and
# denormals; approximations that mirror the CPU (FastExp) use explicit
# intrinsics in the source instead of --use_fast_math.
set(NSS_CUDA_IEEE_OPTIONS -O3 --fmad=true -prec-div=true -prec-sqrt=true -ftz=false)
set(NSS_CUDA_IEEE_SOURCES
  bm3d/kernels.cu
  common/aggregate.cu
  common/match.cu
  nlm/kernels.cu
  runtime/probe.cu
  wnnm/kernels.cu
)
set(NSS_CUDA_CLASSIFIED ${NSS_CUDA_IEEE_SOURCES})
set(_nss_cuda_dedup ${NSS_CUDA_CLASSIFIED})
list(REMOVE_DUPLICATES _nss_cuda_dedup)
if(NOT NSS_CUDA_CLASSIFIED STREQUAL _nss_cuda_dedup)
  message(FATAL_ERROR "CUDA source listed in more than one flag class")
endif()
file(GLOB_RECURSE NSS_CUDA_ALL_CU CONFIGURE_DEPENDS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
set(_nss_cuda_unclassified ${NSS_CUDA_ALL_CU})
list(REMOVE_ITEM _nss_cuda_unclassified ${NSS_CUDA_CLASSIFIED})
if(_nss_cuda_unclassified)
  message(FATAL_ERROR "CUDA sources without a flag-class assignment: ${_nss_cuda_unclassified}")
endif()
set(_nss_cuda_stale ${NSS_CUDA_CLASSIFIED})
list(REMOVE_ITEM _nss_cuda_stale ${NSS_CUDA_ALL_CU})
if(_nss_cuda_stale)
  message(FATAL_ERROR "CUDA flag-class entries without a source file: ${_nss_cuda_stale}")
endif()
set_source_files_properties(${NSS_CUDA_IEEE_SOURCES} PROPERTIES
  COMPILE_OPTIONS "${NSS_CUDA_IEEE_OPTIONS}")

# Host-side runtime (plain C++) shared by every filter.
set(NSS_CUDA_RUNTIME_SOURCES
  runtime/context.cpp
  runtime/device.cpp
  runtime/frame_io.cpp
  runtime/memory.cpp
)

# nss_cuda_core: runtime + all device code, static so tests/cuda can link it.
add_library(nss_cuda_core STATIC ${NSS_CUDA_RUNTIME_SOURCES} ${NSS_CUDA_CLASSIFIED})
set_target_properties(nss_cuda_core PROPERTIES
  CUDA_ARCHITECTURES "${NSS_CUDA_ARCH_LIST}"
  CUDA_RUNTIME_LIBRARY None
  POSITION_INDEPENDENT_CODE ON
  CUDA_VISIBILITY_PRESET hidden
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)
target_include_directories(nss_cuda_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/..")
# Only the (host-compiled) classifier needs the list; nvcc would split a
# comma-separated -D value into several macros.
set_source_files_properties(runtime/device.cpp PROPERTIES COMPILE_DEFINITIONS
  "NSS_CUDA_ARCH_STRING=\"${NSS_CUDA_ARCH_STRING}\";NSS_CUDA_CUBIN_ARCHS=${NSS_CUDA_CUBIN_ARCHS_CSV};NSS_CUDA_PTX_ARCHS=${NSS_CUDA_PTX_ARCHS_CSV}")
target_link_libraries(nss_cuda_core PUBLIC nss_frontend CUDA::cudart_static)

# VapourSynth nodes, one per filter (docs/cuda-plan.md phase order).
add_library(nss_cuda SHARED
  plugin.cpp
  common/group_driver.cpp
  bm3d/filter.cpp
  nlm/filter.cpp
  vaggregate/filter.cpp
  wnnm/filter.cpp
)
set_target_properties(nss_cuda PROPERTIES
  PREFIX "lib"
  OUTPUT_NAME "nss_cuda"
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)
target_compile_definitions(nss_cuda PRIVATE "NSS_VERSION_STRING=\"${NSS_GIT_DESCRIBE}\"")
target_link_libraries(nss_cuda PRIVATE nss_cuda_core VapourSynth::VapourSynth)
