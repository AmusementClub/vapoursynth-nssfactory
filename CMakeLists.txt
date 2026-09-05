cmake_minimum_required(VERSION 3.24)
project(nssfactory LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(NSS_ENABLE_CUDA "Build libnss_cuda.so" OFF)
option(NSS_ENABLE_VULKAN "Build libnss_vk.so" OFF)
option(NSS_BM_KERNEL_LAB "Build isolated BM3D SSD/DCT microkernel experiments" OFF)

set(NSS_HWY_X86 FALSE)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64|i[3-6]86)$" OR
   CMAKE_OSX_ARCHITECTURES MATCHES "^(x86_64|i[3-6]86)$")
  set(NSS_HWY_X86 TRUE)
endif()

# Highway's target list is shared by every CPU TU and by its diagnostic helper.
# The mode is a configure-time choice so target differential builds compile
# only the requested ISA; runtime affinity/dispatcher state cannot change it.
if(DEFINED ENV{NSS_HWY_TARGET_MODE})
  set(NSS_HWY_TARGET_MODE "$ENV{NSS_HWY_TARGET_MODE}")
endif()
set(NSS_HWY_TARGET_MODE "${NSS_HWY_TARGET_MODE}" CACHE STRING
    "Highway target set: dynamic, avx2, or avx3 (defaults to NSS_HWY_TARGET_MODE env)")
if(NOT NSS_HWY_TARGET_MODE)
  set(NSS_HWY_TARGET_MODE "dynamic" CACHE STRING
      "Highway target set: dynamic, avx2, or avx3 (defaults to NSS_HWY_TARGET_MODE env)" FORCE)
endif()
set_property(CACHE NSS_HWY_TARGET_MODE PROPERTY STRINGS dynamic avx2 avx3)
string(TOLOWER "${NSS_HWY_TARGET_MODE}" NSS_HWY_TARGET_MODE)
if(NOT NSS_HWY_TARGET_MODE MATCHES "^(dynamic|avx2|avx3)$")
  message(FATAL_ERROR "NSS_HWY_TARGET_MODE must be dynamic, avx2, or avx3")
endif()
if(NSS_HWY_TARGET_MODE STREQUAL "dynamic")
  # Minimum x86 dispatch policy: AVX2 plus AVX3/AVX3_ZEN4 where available.
  set(NSS_HWY_DISABLED_TARGETS "(HWY_SSE2|HWY_SSSE3|HWY_SSE4|HWY_AVX3_DL|HWY_AVX3_SPR|HWY_AVX10_2)")
elseif(NSS_HWY_TARGET_MODE STREQUAL "avx2")
  set(NSS_HWY_DISABLED_TARGETS
      "(HWY_SSE2|HWY_SSSE3|HWY_SSE4|HWY_AVX3|HWY_AVX3_ZEN4|HWY_AVX3_DL|HWY_AVX3_SPR|HWY_AVX10_2)")
else()
  set(NSS_HWY_DISABLED_TARGETS
      "(HWY_SSE2|HWY_SSSE3|HWY_SSE4|HWY_AVX2|HWY_AVX3_ZEN4|HWY_AVX3_DL|HWY_AVX3_SPR|HWY_AVX10_2)")
endif()
if(NSS_HWY_TARGET_MODE MATCHES "^avx" AND
   NOT NSS_HWY_X86)
  message(FATAL_ERROR "NSS_HWY_TARGET_MODE=${NSS_HWY_TARGET_MODE} requires an x86 processor")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(cmake/Highway.cmake)
find_package(VapourSynth)

set(NSS_GIT_DESCRIBE "" CACHE STRING "Revision recorded in plugin and benchmark metadata")
if(NOT NSS_GIT_DESCRIBE)
  execute_process(
    COMMAND git describe --tags --always --dirty
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_VARIABLE NSS_GIT_DESCRIBE
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
endif()
if(NOT NSS_GIT_DESCRIBE)
  set(NSS_GIT_DESCRIBE "unknown")
endif()

set(NSS_CPU_SOURCES
  src/cpu/batch.cpp
  src/cpu/nlm/nlm.cpp
  src/cpu/bm/ssd.cpp
  src/cpu/bm/spatial4.cpp
  src/cpu/bm/spatial12.cpp
  src/cpu/bm/spatial16.cpp
  src/cpu/bm/dct12.cpp
  src/cpu/bm/dct16.cpp
  src/cpu/bm/dct8.cpp
  src/cpu/bm/match.cpp
  src/cpu/bm/pack.cpp
  src/cpu/bm/agg.cpp
  src/cpu/wnnm/svd_tiny.cpp
  src/cpu/wnnm/svd_batch8.cpp
  src/cpu/wnnm/jacobi8.cpp
  src/cpu/wnnm/shrink.cpp
  src/host/avx2.cpp
  src/host/workspace.cpp
)
file(GLOB NSS_CPU_SECOND_WAVE CONFIGURE_DEPENDS
  src/cpu/common/*.cpp
  src/cpu/mcwnnm/*.cpp
  src/cpu/twsc/*.cpp
  src/cpu/nlh/*.cpp
  src/cpu/ncsr/*.cpp
  src/cpu/lssc/*.cpp
)
list(APPEND NSS_CPU_SOURCES ${NSS_CPU_SECOND_WAVE})
list(APPEND NSS_CPU_SOURCES src/cpu/nlm/distance_horizontal.cpp)
list(APPEND NSS_CPU_SOURCES src/cpu/wnnm/svd_batch8_u_compat.cpp)

add_library(nss_svd_qreplay STATIC src/cpu/wnnm/svd_batch8_qreplay.cpp)
add_library(nss_cpu STATIC ${NSS_CPU_SOURCES})
if(NSS_BM_KERNEL_LAB)
  target_compile_definitions(nss_cpu PRIVATE NSS_BM_KERNEL_LAB=1)
endif()
target_include_directories(nss_svd_qreplay
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_include_directories(nss_cpu
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_link_libraries(nss_svd_qreplay PUBLIC hwy)
target_link_libraries(nss_cpu PUBLIC nss_svd_qreplay hwy)
target_compile_definitions(nss_cpu
  PUBLIC "NSS_VERSION_STRING=\"${NSS_GIT_DESCRIBE}\""
)
set(NSS_CPU_ISA_OPTIONS)
if(NSS_HWY_X86)
  list(APPEND NSS_CPU_ISA_OPTIONS -mavx2 -mfma)
else()
  message(STATUS "NSSFactory: building portable Highway targets for ${CMAKE_SYSTEM_PROCESSOR}")
endif()
target_compile_options(nss_cpu PRIVATE ${NSS_CPU_ISA_OPTIONS})
# Keep the minimum supported x86 target in the build contract as well as in
# the per-TU fallback header.  This prevents a newly added Highway TU from
# silently bringing back pre-AVX2 dispatch targets.
target_compile_definitions(nss_cpu PUBLIC "HWY_DISABLED_TARGETS=${NSS_HWY_DISABLED_TARGETS}")
target_compile_definitions(nss_svd_qreplay PRIVATE "HWY_DISABLED_TARGETS=${NSS_HWY_DISABLED_TARGETS}")
set(NSS_HOT_COMPILE_OPTIONS -O3 -ffast-math)
list(APPEND NSS_HOT_COMPILE_OPTIONS ${NSS_CPU_ISA_OPTIONS})
# Keep IEEE NaN/Inf classification enabled while allowing errno-free hardware
# math and non-trapping vectorization. The plugin does not inspect math errno
# or enable floating-point traps; both options avoid the largest costs of the
# default strict mode without implying -ffinite-math-only.
set(NSS_SAFE_COMPILE_OPTIONS -O3 -fno-math-errno -fno-trapping-math)
list(APPEND NSS_SAFE_COMPILE_OPTIONS ${NSS_CPU_ISA_OPTIONS})
set_source_files_properties(
  src/cpu/nlm/nlm.cpp
  src/cpu/bm/ssd.cpp
  src/cpu/bm/spatial4.cpp
  src/cpu/bm/spatial12.cpp
  src/cpu/bm/spatial16.cpp
  src/cpu/bm/dct12.cpp
  src/cpu/bm/dct16.cpp
  src/cpu/bm/dct8.cpp
  src/cpu/bm/agg.cpp
  src/cpu/bm/pack.cpp
  src/cpu/wnnm/jacobi8.cpp
  src/cpu/wnnm/svd_batch8.cpp
  src/cpu/common/group_center.cpp
  src/cpu/common/iter_regularize.cpp
  src/cpu/common/vec.cpp
  src/cpu/twsc/encode.cpp
  src/cpu/nlh/haar.cpp
  src/cpu/nlh/pixel_match.cpp
  src/cpu/nlh/filter_group.cpp
  src/cpu/mcwnnm/match_nch.cpp
  PROPERTIES COMPILE_OPTIONS "${NSS_HOT_COMPILE_OPTIONS}"
)
set(NSS_FINITE_HOT_COMPILE_OPTIONS ${NSS_HOT_COMPILE_OPTIONS})
list(APPEND NSS_FINITE_HOT_COMPILE_OPTIONS -fno-finite-math-only)
set(NSS_MATCHER_COMPILE_OPTIONS ${NSS_FINITE_HOT_COMPILE_OPTIONS})
list(APPEND NSS_MATCHER_COMPILE_OPTIONS -fvisibility=hidden)
set_source_files_properties(
  src/cpu/nlh/match16.cpp
  PROPERTIES COMPILE_OPTIONS "${NSS_MATCHER_COMPILE_OPTIONS}"
)
set_source_files_properties(
  src/cpu/nlm/distance_horizontal.cpp
  PROPERTIES COMPILE_OPTIONS "${NSS_MATCHER_COMPILE_OPTIONS}"
)
set(NSS_SAFE_HIDDEN_COMPILE_OPTIONS ${NSS_SAFE_COMPILE_OPTIONS})
list(APPEND NSS_SAFE_HIDDEN_COMPILE_OPTIONS -fvisibility=hidden)
set_source_files_properties(
  src/cpu/wnnm/svd_batch8_u_compat.cpp
  PROPERTIES COMPILE_OPTIONS "${NSS_SAFE_HIDDEN_COMPILE_OPTIONS}"
)
set(NSS_HOT_HIDDEN_COMPILE_OPTIONS ${NSS_HOT_COMPILE_OPTIONS})
list(APPEND NSS_HOT_HIDDEN_COMPILE_OPTIONS -fvisibility=hidden)
set_source_files_properties(
  src/cpu/wnnm/svd_batch8_qreplay.cpp
  PROPERTIES COMPILE_OPTIONS "${NSS_HOT_HIDDEN_COMPILE_OPTIONS}"
)
set_source_files_properties(
  src/cpu/common/soft_threshold.cpp
  src/cpu/common/admm_x.cpp
  src/cpu/mcwnnm/admm.cpp
  src/cpu/ncsr/centralize.cpp
  src/cpu/lssc/reconstruct.cpp
  PROPERTIES COMPILE_OPTIONS "${NSS_SAFE_COMPILE_OPTIONS}"
)
if(VapourSynth_FOUND)
  file(GLOB NSS_HOST_FILTERS CONFIGURE_DEPENDS src/host/filter_*.cpp)
  add_library(nss SHARED
    src/host/plugin.cpp
    src/host/validate.cpp
    ${NSS_HOST_FILTERS}
  )
  set_target_properties(nss PROPERTIES
    PREFIX "lib"
    OUTPUT_NAME "nss"
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
  )
  target_include_directories(nss PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
  )
  target_link_libraries(nss PRIVATE nss_cpu VapourSynth::VapourSynth)
endif()

enable_testing()
add_subdirectory(tests)
