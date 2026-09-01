cmake_minimum_required(VERSION 3.24)
project(nssfactory LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(NSS_ENABLE_CUDA "Build libnss_cuda.so" OFF)
option(NSS_ENABLE_VULKAN "Build libnss_vk.so" OFF)

# Highway's target list is shared by every CPU TU and by its diagnostic helper.
# Keep the minimum x86 ISA in one CMake-visible contract so both report the
# same generated dispatch set.
set(NSS_HWY_DISABLED_TARGETS "(HWY_SSE2|HWY_SSSE3|HWY_SSE4|HWY_AVX3_DL|HWY_AVX3_SPR|HWY_AVX10_2)")

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

add_library(nss_cpu STATIC ${NSS_CPU_SOURCES})
target_include_directories(nss_cpu
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_link_libraries(nss_cpu PUBLIC hwy)
target_compile_definitions(nss_cpu
  PUBLIC "NSS_VERSION_STRING=\"${NSS_GIT_DESCRIBE}\""
)
set(NSS_CPU_ISA_OPTIONS)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64|i[3-6]86)$")
  list(APPEND NSS_CPU_ISA_OPTIONS -mavx2 -mfma)
else()
  message(STATUS "NSSFactory: building portable Highway targets for ${CMAKE_SYSTEM_PROCESSOR}")
endif()
target_compile_options(nss_cpu PRIVATE ${NSS_CPU_ISA_OPTIONS})
# Keep the minimum supported x86 target in the build contract as well as in
# the per-TU fallback header.  This prevents a newly added Highway TU from
# silently bringing back pre-AVX2 dispatch targets.
target_compile_definitions(nss_cpu PRIVATE "HWY_DISABLED_TARGETS=${NSS_HWY_DISABLED_TARGETS}")
set(NSS_HOT_COMPILE_OPTIONS -O3 -ffast-math)
list(APPEND NSS_HOT_COMPILE_OPTIONS ${NSS_CPU_ISA_OPTIONS})
set_source_files_properties(
  src/cpu/nlm/nlm.cpp
  src/cpu/bm/ssd.cpp
  src/cpu/bm/dct8.cpp
  src/cpu/bm/agg.cpp
  src/cpu/bm/pack.cpp
  src/cpu/wnnm/jacobi8.cpp
  src/cpu/wnnm/svd_batch8.cpp
  src/cpu/common/group_center.cpp
  src/cpu/common/soft_threshold.cpp
  src/cpu/common/iter_regularize.cpp
  src/cpu/common/admm_x.cpp
  src/cpu/common/vec.cpp
  src/cpu/mcwnnm/admm.cpp
  src/cpu/twsc/encode.cpp
  src/cpu/nlh/haar.cpp
  src/cpu/nlh/pixel_match.cpp
  src/cpu/nlh/filter_group.cpp
  src/cpu/ncsr/centralize.cpp
  src/cpu/lssc/reconstruct.cpp
  PROPERTIES COMPILE_OPTIONS "${NSS_HOT_COMPILE_OPTIONS}"
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
