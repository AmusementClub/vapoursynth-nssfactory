cmake_minimum_required(VERSION 3.24)
project(nssfactory LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(NSS_ENABLE_CUDA "Build libnss_cuda.so" OFF)
option(NSS_ENABLE_VULKAN "Build libnss_vk.so" OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(cmake/Highway.cmake)
find_package(VapourSynth)

set(NSS_GIT_DESCRIBE "unknown")
execute_process(
  COMMAND git describe --tags --always --dirty
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  OUTPUT_VARIABLE NSS_GIT_DESCRIBE
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)
if(NOT NSS_GIT_DESCRIBE)
  set(NSS_GIT_DESCRIBE "unknown")
endif()

set(NSS_CPU_SOURCES
  src/cpu/nlm/nlm.cpp
  src/cpu/bm/ssd.cpp
  src/cpu/bm/dct8.cpp
  src/cpu/bm/match.cpp
  src/cpu/bm/pack.cpp
  src/cpu/bm/agg.cpp
  src/cpu/wnnm/svd_tiny.cpp
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
target_compile_options(nss_cpu PRIVATE -mavx2 -mfma)
set_source_files_properties(
  src/cpu/nlm/nlm.cpp
  src/cpu/bm/ssd.cpp
  src/cpu/bm/dct8.cpp
  src/cpu/bm/agg.cpp
  src/cpu/bm/pack.cpp
  src/cpu/wnnm/jacobi8.cpp
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
  PROPERTIES COMPILE_OPTIONS "-O3;-ffast-math;-mavx2;-mfma"
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
