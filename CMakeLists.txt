# SPDX-License-Identifier: GPL-2.0-only
cmake_minimum_required(VERSION 3.24)
project(nssfactory LANGUAGES CXX)

# Single-config generators default to Release: the hot driver TUs carry no
# per-file -O3, so a bare `cmake -B build` must not produce an -O0 library.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  message(STATUS "NSSFactory: CMAKE_BUILD_TYPE unset, defaulting to Release")
endif()

set(NSS_HWY_X86 FALSE)
set(NSS_HWY_ARM64 FALSE)
# A single explicit macOS architecture overrides the native host architecture.
set(NSS_TARGET_PROCESSOR "${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_OSX_ARCHITECTURES)
  if(CMAKE_OSX_ARCHITECTURES MATCHES ";")
    message(FATAL_ERROR "Build and validate each macOS architecture separately before combining binaries")
  endif()
  set(NSS_TARGET_PROCESSOR "${CMAKE_OSX_ARCHITECTURES}")
endif()
if(NSS_TARGET_PROCESSOR MATCHES "^(x86_64|amd64|AMD64|i[3-6]86)$")
  set(NSS_HWY_X86 TRUE)
elseif(NSS_TARGET_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
  set(NSS_HWY_ARM64 TRUE)
endif()

# Independent port candidates; new vector paths are restricted to Highway AVX2.
set(NSS_AVX2_EXPERIMENT "0" CACHE STRING "Additional AVX2 candidate bits; use NSS_AVX2_DEFAULTS=OFF and zero for the baseline")
if(NOT NSS_AVX2_EXPERIMENT MATCHES "^[0-9]+$" OR NSS_AVX2_EXPERIMENT GREATER 2047)
  message(FATAL_ERROR "NSS_AVX2_EXPERIMENT must be a bitmask in 0..2047")
endif()
option(NSS_AVX2_DEFAULTS "Enable configuration-scoped, dual-host validated AVX2 paths" ${NSS_HWY_X86})
if(NSS_HWY_ARM64 AND (NSS_AVX2_DEFAULTS OR NOT NSS_AVX2_EXPERIMENT STREQUAL "0"))
  message(FATAL_ERROR "ARM baseline requires NSS_AVX2_DEFAULTS=OFF and NSS_AVX2_EXPERIMENT=0")
endif()
# Only these kernels are compiled for the configuration admission policy.
# Experimental masks are independent and bypass configuration admission.
set(NSS_AVX2_DEFAULT_MASK 1789)
if(NSS_AVX2_DEFAULTS)
  math(EXPR NSS_AVX2_COMPILED "${NSS_AVX2_EXPERIMENT} | ${NSS_AVX2_DEFAULT_MASK}")
else()
  set(NSS_AVX2_COMPILED ${NSS_AVX2_EXPERIMENT})
endif()
add_compile_definitions(NSS_AVX2_REQUESTED=${NSS_AVX2_EXPERIMENT}
                        NSS_AVX2_DEFAULTS=$<BOOL:${NSS_AVX2_DEFAULTS}>)
# Scope definitions to participating translation units so independent candidate
# builds do not rebuild unrelated filters or alter their compiler inputs.
foreach(entry "bm/ssd.cpp|1" "bm/spatial4.cpp|2" "bm/spatial12.cpp|4"
              "bm/spatial16.cpp|8" "nlh/haar.cpp|32"
              "bm/dct16.cpp|128" "nlh/match16.cpp|256" "bm/pack.cpp|512"
              "wnnm/jacobi8.cpp|1024")
  string(REPLACE "|" ";" fields "${entry}")
  list(GET fields 0 file)
  list(GET fields 1 bits)
  math(EXPR local_mask "${NSS_AVX2_COMPILED} & ${bits}")
  set_property(SOURCE "src/cpu/${file}" APPEND PROPERTY COMPILE_DEFINITIONS "NSS_AVX2_EXPERIMENT=${local_mask}")
endforeach()

# Selected C4 combination: SortedTopK (1), rolling ring/scratch (256), reuse (2048).
# Zero retains the pure correctness reference; other masks remain selectable.
set(NSS_BM_EXPERIMENT "2305" CACHE STRING "BM3D feature bitmask; 2305 is the selected default, zero is the reference")
if(NOT NSS_BM_EXPERIMENT MATCHES "^[0-9]+$")
  message(FATAL_ERROR "NSS_BM_EXPERIMENT must be an unsigned bitmask")
endif()
if(NSS_BM_EXPERIMENT GREATER 65535)
  message(FATAL_ERROR "Unsupported BM3D experiment bits")
endif()
math(EXPR NSS_BM_TOPK_BITS "${NSS_BM_EXPERIMENT} & 3")
if(NSS_BM_TOPK_BITS EQUAL 3)
  message(FATAL_ERROR "Select either sorted or cached-worst top-k")
endif()
add_compile_definitions(NSS_BM_EXPERIMENT=${NSS_BM_EXPERIMENT})
foreach(feature REUSE RASTER HOMOGENEOUS SCRATCH RING)
  if(feature STREQUAL "REUSE")
    set(bits 2052)
  elseif(feature STREQUAL "RASTER")
    set(bits 4100)
  elseif(feature STREQUAL "HOMOGENEOUS")
    set(bits 8196)
  elseif(feature STREQUAL "SCRATCH")
    set(bits 16640)
  else()
    set(bits 33024)
  endif()
  math(EXPR enabled "${NSS_BM_EXPERIMENT} & ${bits}")
  add_compile_definitions(NSS_BM_${feature}=${enabled})
endforeach()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(NSS_BM_KERNEL_LAB "Build isolated BM3D SSD/DCT microkernel experiments" OFF)
option(NSS_LSSC_EXPLORER "Build the isolated paper-track LSSC exploration library" OFF)
option(NSS_TWSC_MIDGROUP_BATCH "Enable the FP64 G24/G32 TWSC batch path" ${NSS_HWY_X86})
option(NSS_GEMM_MULTI_ACCUM "Enable the opt-in exact GEMM multi-accumulator candidate" OFF)
option(NSS_ENABLE_SVD_LIBRARY "Build the opt-in LAPACKE DGESDD TWSC adapter" OFF)
set(NSS_SVD_LIBRARY_PROVIDER_VERSION "unknown" CACHE STRING
  "Provider version recorded in the optional SVD manifest")
add_compile_definitions(NSS_TWSC_MIDGROUP_BATCH=$<BOOL:${NSS_TWSC_MIDGROUP_BATCH}>
                        NSS_GEMM_MULTI_ACCUM=$<BOOL:${NSS_GEMM_MULTI_ACCUM}>)

# Highway's target list is shared by every CPU TU and by its diagnostic helper.
# The mode is a configure-time choice so target differential builds compile
# only the requested ISA; runtime affinity/dispatcher state cannot change it.
if(DEFINED ENV{NSS_HWY_TARGET_MODE})
  set(NSS_HWY_TARGET_MODE "$ENV{NSS_HWY_TARGET_MODE}")
endif()
set(NSS_HWY_TARGET_MODE "${NSS_HWY_TARGET_MODE}" CACHE STRING
    "Highway target set: dynamic, avx2, avx3, neon, or portable-test")
if(NOT NSS_HWY_TARGET_MODE)
  set(NSS_HWY_TARGET_MODE "dynamic" CACHE STRING
      "Highway target set: dynamic, avx2, avx3, neon, or portable-test" FORCE)
endif()
set_property(CACHE NSS_HWY_TARGET_MODE PROPERTY STRINGS dynamic avx2 avx3 neon portable-test)
string(TOLOWER "${NSS_HWY_TARGET_MODE}" NSS_HWY_TARGET_MODE)
if(NOT NSS_HWY_TARGET_MODE MATCHES "^(dynamic|avx2|avx3|neon|portable-test)$")
  message(FATAL_ERROR "NSS_HWY_TARGET_MODE must be dynamic, avx2, avx3, neon, or portable-test")
endif()
if(NSS_HWY_TARGET_MODE MATCHES "^(neon|portable-test)$" AND NOT NSS_HWY_ARM64)
  message(FATAL_ERROR "NSS_HWY_TARGET_MODE=${NSS_HWY_TARGET_MODE} requires AArch64")
endif()
set(NSS_HWY_EXTRA_DEFINITIONS)
if(NSS_HWY_ARM64)
  # Pin the first ARM route to the ordinary NEON family. In particular, SVE
  # sizeless vector arrays and BF16 variants have not cleared this port's gate.
  set(NSS_HWY_DISABLED_TARGETS "(HWY_ALL_SVE|HWY_NEON_BF16)")
  if(NSS_HWY_TARGET_MODE STREQUAL "portable-test")
    list(APPEND NSS_HWY_EXTRA_DEFINITIONS HWY_COMPILE_ONLY_EMU128=1)
  endif()
elseif(NSS_HWY_TARGET_MODE STREQUAL "dynamic")
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
if(NSS_HWY_TARGET_MODE STREQUAL "portable-test")
  # Highway disables EMU128 on affected compilers (including GCC 13). Fail
  # during configuration instead of silently building a different oracle.
  include(CheckCXXSourceCompiles)
  set(CMAKE_REQUIRED_INCLUDES "${highway_SOURCE_DIR}")
  set(CMAKE_REQUIRED_DEFINITIONS -DHWY_COMPILE_ONLY_EMU128=1)
  check_cxx_source_compiles("#include <hwy/detect_targets.h>
    #if HWY_TARGETS != HWY_EMU128
    #error EMU128 is unavailable for this compiler
    #endif
    int main() { return 0; }" NSS_HAS_EMU128)
  unset(CMAKE_REQUIRED_INCLUDES)
  unset(CMAKE_REQUIRED_DEFINITIONS)
  if(NOT NSS_HAS_EMU128)
    message(FATAL_ERROR "portable-test requires a Highway-supported EMU128 compiler; use Clang instead of overriding Highway's compiler blocklist")
  endif()
endif()
find_package(VapourSynth)

# clang-cl parses the MSVC command-line dialect and silently IGNORES GNU-style
# flags (observed: -ffast-math/-fno-fast-math/-ffp-contract=off/-O3 dropped with
# -Wunknown-argument), which voids the flag-class numerical contract. Forward
# every GNU-style flag through /clang: on that driver; GNU/Clang drivers use
# the lists unchanged. -fvisibility=hidden is dropped (default on Windows).
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
  set(NSS_CLANG_CL ON)
else()
  set(NSS_CLANG_CL OFF)
endif()
function(nss_cc_flags out)
  if(NSS_CLANG_CL)
    set(translated "")
    foreach(flag ${ARGN})
      if(NOT flag STREQUAL "-fvisibility=hidden")
        list(APPEND translated "/clang:${flag}")
      endif()
    endforeach()
    set(${out} ${translated} PARENT_SCOPE)
  else()
    set(${out} ${ARGN} PARENT_SCOPE)
  endif()
endfunction()
nss_cc_flags(NSS_SAFE_PIN_COMPILE_OPTIONS -O3 -fno-fast-math -ffp-contract=off)

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
  src/cpu/backend.cpp
  src/cpu/batch.cpp
  src/cpu/nlm/nlm.cpp
  src/cpu/bm/ssd.cpp
  src/cpu/bm/match_joint.cpp
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
file(GLOB NSS_CPU_SECOND_WAVE CONFIGURE_DEPENDS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  src/cpu/common/*.cpp
  src/cpu/mcwnnm/*.cpp
  src/cpu/twsc/*.cpp
  src/cpu/nlh/*.cpp
  src/cpu/ncsr/*.cpp
  src/cpu/lssc/*.cpp
)
list(FILTER NSS_CPU_SECOND_WAVE EXCLUDE REGEX "/lssc/gemm_sme\\.cpp$")
list(FILTER NSS_CPU_SECOND_WAVE EXCLUDE REGEX "/twsc/svd_library\\.cpp$")
option(NSS_ENABLE_LSSC_SME "Build the optional LSSC SME matrix leaf when supported" OFF)
set(NSS_HAVE_LSSC_SME 0)
if(NSS_ENABLE_LSSC_SME AND APPLE AND NSS_HWY_ARM64 AND
   CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NOT NSS_HWY_TARGET_MODE STREQUAL "portable-test")
  include(CheckCXXSourceCompiles)
  include(CMakePushCheckState)
  cmake_push_check_state(RESET)
  set(CMAKE_REQUIRED_FLAGS "-march=armv8.6-a+sme")
  check_cxx_source_compiles("#include <arm_sme.h>
    #include <arm_sve.h>
    extern \"C\" __arm_new(\"za\") __arm_locally_streaming
    void probe() { svzero_za(); }
    int main() { return 0; }" NSS_COMPILER_HAS_LSSC_SME)
  cmake_pop_check_state()
  if(NSS_COMPILER_HAS_LSSC_SME)
    set(NSS_HAVE_LSSC_SME 1)
    list(APPEND NSS_CPU_SECOND_WAVE src/cpu/lssc/gemm_sme.cpp)
    set_source_files_properties(src/cpu/lssc/gemm_sme.cpp PROPERTIES
      COMPILE_OPTIONS "-O3;-march=armv8.6-a+sme")
  endif()
endif()
message(STATUS "NSSFactory: optional LSSC SME leaf compiled=${NSS_HAVE_LSSC_SME}")
list(APPEND NSS_CPU_SOURCES ${NSS_CPU_SECOND_WAVE})
list(APPEND NSS_CPU_SOURCES src/cpu/nlm/distance_horizontal.cpp)
list(APPEND NSS_CPU_SOURCES src/cpu/wnnm/svd_batch8_u_compat.cpp)

# The LAPACKE adapter is intentionally a separate target during the first
# milestone. The default graph never sees provider headers or symbols, and
# unrelated nss_cpu consumers do not inherit an OpenBLAS/LAPACK link.
if(NSS_ENABLE_SVD_LIBRARY)
  if(NOT NSS_HWY_X86)
    message(FATAL_ERROR "NSS_ENABLE_SVD_LIBRARY is initially supported only on x86; ARM/NEON remains provider-off")
  endif()
  include(CheckCXXSourceCompiles)
  include(CMakePushCheckState)
  find_path(NSS_SVD_LAPACKE_INCLUDE_DIR lapacke.h)
  find_library(NSS_SVD_LAPACKE_LIBRARY NAMES lapacke liblapacke)
  find_library(NSS_SVD_LAPACK_LIBRARY NAMES openblas lapack libopenblas)
  find_library(NSS_SVD_BLAS_LIBRARY NAMES openblas blas libopenblas)
  if(NOT NSS_SVD_LAPACKE_INCLUDE_DIR OR NOT NSS_SVD_LAPACKE_LIBRARY OR
     NOT NSS_SVD_LAPACK_LIBRARY OR NOT NSS_SVD_BLAS_LIBRARY)
    message(FATAL_ERROR
      "NSS_ENABLE_SVD_LIBRARY=ON requires an LP64 LAPACKE provider (lapacke.h, "
      "liblapacke, LAPACK/OpenBLAS, and BLAS/OpenBLAS libraries)")
  endif()
  cmake_push_check_state(RESET)
  set(CMAKE_REQUIRED_INCLUDES "${NSS_SVD_LAPACKE_INCLUDE_DIR}")
  set(CMAKE_REQUIRED_LIBRARIES "${NSS_SVD_LAPACKE_LIBRARY};${NSS_SVD_LAPACK_LIBRARY};${NSS_SVD_BLAS_LIBRARY}")
  check_cxx_source_compiles([=[
    #include <lapacke.h>
    #include <type_traits>
    static_assert(sizeof(lapack_int) == 4, "the SVD adapter requires LP64 lapack_int");
    int main() {
      double a[1] = {0}, s[1] = {0}, u[1] = {0}, vt[1] = {0}, work[1] = {0};
      lapack_int iwork[8] = {}, info = LAPACKE_dgesdd_work(
        LAPACK_COL_MAJOR, 'S', 1, 1, a, 1, s, u, 1, vt, 1, work, 1, iwork);
      return static_cast<int>(info);
    }
  ]=] NSS_SVD_LAPACKE_ABI_OK)
  cmake_pop_check_state()
  if(NOT NSS_SVD_LAPACKE_ABI_OK)
    message(FATAL_ERROR "The selected LAPACKE provider failed the LP64 DGESDD column-major ABI probe")
  endif()
  add_library(nss_svd_provider INTERFACE)
  target_include_directories(nss_svd_provider INTERFACE "${NSS_SVD_LAPACKE_INCLUDE_DIR}")
  target_link_libraries(nss_svd_provider INTERFACE
    "${NSS_SVD_LAPACKE_LIBRARY}" "${NSS_SVD_LAPACK_LIBRARY}" "${NSS_SVD_BLAS_LIBRARY}")
  add_library(nss_svd_library STATIC src/cpu/twsc/svd_library.cpp)
  target_include_directories(nss_svd_library
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_link_libraries(nss_svd_library PRIVATE nss_svd_provider)
  target_compile_definitions(nss_svd_library PRIVATE NSS_ENABLE_SVD_LIBRARY=1)
  set_source_files_properties(src/cpu/twsc/svd_library.cpp PROPERTIES
    COMPILE_OPTIONS "${NSS_SAFE_PIN_COMPILE_OPTIONS}")

  get_filename_component(NSS_SVD_LAPACKE_LIBRARY_REAL "${NSS_SVD_LAPACKE_LIBRARY}" REALPATH)
  file(SHA256 "${NSS_SVD_LAPACKE_LIBRARY_REAL}" NSS_SVD_LAPACKE_LIBRARY_SHA256)
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/nss-svd-library-manifest.json"
    "{\n"
    "  \"provider\": \"LAPACKE/DGESDD\",\n"
    "  \"provider_version\": \"${NSS_SVD_LIBRARY_PROVIDER_VERSION}\",\n"
    "  \"lapack_int_bytes\": 4,\n"
    "  \"lapacke_library\": \"${NSS_SVD_LAPACKE_LIBRARY_REAL}\",\n"
    "  \"lapacke_sha256\": \"${NSS_SVD_LAPACKE_LIBRARY_SHA256}\",\n"
    "  \"thread_policy\": \"provider configuration must be single-threaded with locking\",\n"
    "  \"target_isa\": \"${NSS_HWY_TARGET_MODE}\",\n"
    "  \"automatic_dispatch\": false\n"
    "}\n")
endif()

add_library(nss_svd_qreplay STATIC src/cpu/wnnm/svd_batch8_qreplay.cpp)
add_library(nss_cpu STATIC ${NSS_CPU_SOURCES})
# The plugin's public surface is the VapourSynth entry point only; CPU
# kernels must not leak as interposable dynamic symbols from libnss. Static
# archives keep these symbols usable to statically linked test executables.
set_target_properties(nss_cpu nss_svd_qreplay PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON)
option(NSS_ALIGNMENT_GENERIC "Use general TWSC/NLH math and distance controls for same-model validation" OFF)
target_compile_definitions(nss_cpu PRIVATE NSS_ALIGNMENT_GENERIC=$<BOOL:${NSS_ALIGNMENT_GENERIC}>)
target_compile_definitions(nss_cpu PRIVATE NSS_HAVE_LSSC_SME=${NSS_HAVE_LSSC_SME})
if(NSS_BM_KERNEL_LAB)
  target_compile_definitions(nss_cpu PRIVATE NSS_BM_KERNEL_LAB=1)
  # Internal campaign selectors affect only the experimental translation unit.
  # Ordinary builds contain neither the selectors nor their candidate kernels.
  foreach(pair "DCT8;dct8" "DCT12;dct12" "MATCH8;ssd" "FFTW;dct8")
    list(GET pair 0 kernel)
    list(GET pair 1 source)
    if(DEFINED NSS_LAB_${kernel}_VARIANT)
      if(NOT NSS_LAB_${kernel}_VARIANT MATCHES "^[12]$" OR
         (kernel STREQUAL "FFTW" AND NOT NSS_LAB_${kernel}_VARIANT STREQUAL "1"))
        message(FATAL_ERROR "Invalid NSS_LAB_${kernel}_VARIANT")
      endif()
      set_property(SOURCE src/cpu/bm/${source}.cpp APPEND PROPERTY COMPILE_DEFINITIONS
        "NSS_LAB_${kernel}_VARIANT=${NSS_LAB_${kernel}_VARIANT}")
    endif()
  endforeach()
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
nss_cc_flags(NSS_CPU_ISA_OPTIONS_CC ${NSS_CPU_ISA_OPTIONS})
target_compile_options(nss_cpu PRIVATE ${NSS_CPU_ISA_OPTIONS_CC})
# Keep the minimum supported x86 target in the build contract as well as in
# the per-TU fallback header.  This prevents a newly added Highway TU from
# silently bringing back pre-AVX2 dispatch targets.
target_compile_definitions(nss_cpu PUBLIC "HWY_DISABLED_TARGETS=${NSS_HWY_DISABLED_TARGETS}")
target_compile_definitions(nss_svd_qreplay PRIVATE "HWY_DISABLED_TARGETS=${NSS_HWY_DISABLED_TARGETS}")
target_compile_definitions(nss_cpu PUBLIC ${NSS_HWY_EXTRA_DEFINITIONS}
  "NSS_HWY_TARGET_MODE_STRING=\"${NSS_HWY_TARGET_MODE}\"")
target_compile_definitions(nss_svd_qreplay PRIVATE ${NSS_HWY_EXTRA_DEFINITIONS})
set(NSS_HOT_COMPILE_OPTIONS -O3 -ffast-math)
list(APPEND NSS_HOT_COMPILE_OPTIONS ${NSS_CPU_ISA_OPTIONS})
# Keep IEEE NaN/Inf classification enabled while allowing errno-free hardware
# math and non-trapping vectorization. The plugin does not inspect math errno
# or enable floating-point traps; both options avoid the largest costs of the
# default strict mode without implying -ffinite-math-only.
set(NSS_SAFE_COMPILE_OPTIONS -O3 -fno-math-errno -fno-trapping-math)
list(APPEND NSS_SAFE_COMPILE_OPTIONS ${NSS_CPU_ISA_OPTIONS})
set(NSS_FINITE_HOT_COMPILE_OPTIONS ${NSS_HOT_COMPILE_OPTIONS})
list(APPEND NSS_FINITE_HOT_COMPILE_OPTIONS -fno-finite-math-only)
set(NSS_MATCHER_COMPILE_OPTIONS ${NSS_FINITE_HOT_COMPILE_OPTIONS})
list(APPEND NSS_MATCHER_COMPILE_OPTIONS -fvisibility=hidden)
set(NSS_SAFE_HIDDEN_COMPILE_OPTIONS ${NSS_SAFE_COMPILE_OPTIONS})
list(APPEND NSS_SAFE_HIDDEN_COMPILE_OPTIONS -fvisibility=hidden)
set(NSS_HOT_HIDDEN_COMPILE_OPTIONS ${NSS_HOT_COMPILE_OPTIONS})
list(APPEND NSS_HOT_HIDDEN_COMPILE_OPTIONS -fvisibility=hidden)

# Canonical lists stay GNU-style for the guardrails below; use sites take the
# clang-cl-translated copies.
nss_cc_flags(NSS_HOT_COMPILE_OPTIONS_CC ${NSS_HOT_COMPILE_OPTIONS})
nss_cc_flags(NSS_SAFE_COMPILE_OPTIONS_CC ${NSS_SAFE_COMPILE_OPTIONS})
nss_cc_flags(NSS_FINITE_HOT_COMPILE_OPTIONS_CC ${NSS_FINITE_HOT_COMPILE_OPTIONS})
nss_cc_flags(NSS_MATCHER_COMPILE_OPTIONS_CC ${NSS_MATCHER_COMPILE_OPTIONS})
nss_cc_flags(NSS_SAFE_HIDDEN_COMPILE_OPTIONS_CC ${NSS_SAFE_HIDDEN_COMPILE_OPTIONS})
nss_cc_flags(NSS_HOT_HIDDEN_COMPILE_OPTIONS_CC ${NSS_HOT_HIDDEN_COMPILE_OPTIONS})

# Flag-class membership is a whitelist: every CPU translation unit belongs to
# exactly one class below, or to NSS_DEFAULT_CPU_SOURCES (plain target-level
# options). The validation block after the class definitions turns a missing
# or duplicate assignment into a configure-time error, so a new globbed TU
# can never silently land in the default class.
set(NSS_PINNED_SOURCES
  src/cpu/bm/match_joint.cpp
  src/cpu/common/gemm_precise.cpp
  src/cpu/twsc/linalg.cpp
  src/cpu/twsc/svd_batch.cpp
  src/cpu/nlh/fused.cpp
)
set_source_files_properties(${NSS_PINNED_SOURCES}
  PROPERTIES COMPILE_OPTIONS "${NSS_SAFE_PIN_COMPILE_OPTIONS}")
set(NSS_HOT_SOURCES
  src/cpu/nlm/nlm.cpp
  src/cpu/bm/ssd.cpp
  src/cpu/common/image_ssd_row.cpp
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
  src/cpu/mcwnnm/match_nch.cpp
)
set_source_files_properties(${NSS_HOT_SOURCES}
  PROPERTIES COMPILE_OPTIONS "${NSS_HOT_COMPILE_OPTIONS_CC}")
# LSSC k-means step kernels compare computed distances like the matchers do;
# keep IEEE comparison semantics (-fno-finite-math-only) while the SSD
# arithmetic itself matches the vec.cpp HOT class bit-for-bit.
set(NSS_FINITE_HOT_SOURCES
  src/cpu/lssc/cluster_ssd.cpp
)
set_source_files_properties(${NSS_FINITE_HOT_SOURCES}
  PROPERTIES COMPILE_OPTIONS "${NSS_FINITE_HOT_COMPILE_OPTIONS_CC}")
set(NSS_MATCHER_SOURCES
  src/cpu/nlh/match16.cpp
  src/cpu/nlm/distance_horizontal.cpp
)
set_source_files_properties(${NSS_MATCHER_SOURCES}
  PROPERTIES COMPILE_OPTIONS "${NSS_MATCHER_COMPILE_OPTIONS_CC}")
set(NSS_SAFE_HIDDEN_SOURCES
  src/cpu/wnnm/svd_batch8_u_compat.cpp
)
set_source_files_properties(${NSS_SAFE_HIDDEN_SOURCES}
  PROPERTIES COMPILE_OPTIONS "${NSS_SAFE_HIDDEN_COMPILE_OPTIONS_CC}")
set(NSS_HOT_HIDDEN_SOURCES
  src/cpu/wnnm/svd_batch8_qreplay.cpp
)
set_source_files_properties(${NSS_HOT_HIDDEN_SOURCES}
  PROPERTIES COMPILE_OPTIONS "${NSS_HOT_HIDDEN_COMPILE_OPTIONS_CC}")
set(NSS_SAFE_SOURCES
  src/cpu/common/soft_threshold.cpp
  src/cpu/common/admm_x.cpp
  src/cpu/mcwnnm/admm.cpp
  src/cpu/ncsr/centralize.cpp
  src/cpu/lssc/reconstruct.cpp
)
set_source_files_properties(${NSS_SAFE_SOURCES}
  PROPERTIES COMPILE_OPTIONS "${NSS_SAFE_COMPILE_OPTIONS_CC}")

# CPU TUs deliberately built with plain target-level options (no per-file
# flags). Keep this list explicit so the default class is a decision, not a
# fallback.
set(NSS_DEFAULT_CPU_SOURCES
  src/cpu/backend.cpp
  src/cpu/batch.cpp
  src/cpu/bm/match.cpp
  src/cpu/wnnm/svd_tiny.cpp
  src/cpu/wnnm/shrink.cpp
  src/cpu/common/image.cpp
  src/cpu/common/pack_nch.cpp
  src/cpu/common/sv_shrink.cpp
  src/cpu/mcwnnm/filter_group.cpp
  src/cpu/twsc/batch_full.cpp
  src/cpu/twsc/full.cpp
  src/cpu/twsc/image.cpp
  src/cpu/twsc/math.cpp
  src/cpu/nlh/full.cpp
  src/cpu/nlh/image.cpp
  src/cpu/ncsr/hq.cpp
  src/cpu/lssc/cluster.cpp
  src/cpu/lssc/dict.cpp
  src/cpu/lssc/gemm.cpp
  src/cpu/lssc/omp.cpp
)

# Whitelist validation. gemm_sme.cpp carries its own conditional options and
# svd_batch8_qreplay.cpp builds into nss_svd_qreplay; host leaves are not CPU
# kernels.
set(NSS_CLASSIFIED_CPU_SOURCES
  ${NSS_PINNED_SOURCES} ${NSS_HOT_SOURCES} ${NSS_FINITE_HOT_SOURCES}
  ${NSS_MATCHER_SOURCES} ${NSS_SAFE_HIDDEN_SOURCES} ${NSS_HOT_HIDDEN_SOURCES}
  ${NSS_SAFE_SOURCES})
if(NSS_HAVE_LSSC_SME)
  list(APPEND NSS_CLASSIFIED_CPU_SOURCES src/cpu/lssc/gemm_sme.cpp)
endif()
set(_nss_class_dedup ${NSS_CLASSIFIED_CPU_SOURCES})
list(REMOVE_DUPLICATES _nss_class_dedup)
list(LENGTH NSS_CLASSIFIED_CPU_SOURCES _nss_class_count)
list(LENGTH _nss_class_dedup _nss_class_dedup_count)
if(NOT _nss_class_count EQUAL _nss_class_dedup_count)
  message(FATAL_ERROR "CPU source listed in more than one flag class")
endif()
set(_nss_unclassified ${NSS_CPU_SOURCES})
list(REMOVE_ITEM _nss_unclassified
  ${NSS_CLASSIFIED_CPU_SOURCES} ${NSS_DEFAULT_CPU_SOURCES}
  src/host/avx2.cpp src/host/workspace.cpp)
if(_nss_unclassified)
  message(FATAL_ERROR "CPU sources without a flag-class assignment (add to a class or NSS_DEFAULT_CPU_SOURCES): ${_nss_unclassified}")
endif()
set(_nss_stale ${NSS_CLASSIFIED_CPU_SOURCES} ${NSS_DEFAULT_CPU_SOURCES})
list(REMOVE_ITEM _nss_stale ${NSS_CPU_SOURCES} src/cpu/wnnm/svd_batch8_qreplay.cpp)
if(_nss_stale)
  message(FATAL_ERROR "flag-class entries not present in nss_cpu sources: ${_nss_stale}")
endif()
set(_nss_finite_hot_expected ${NSS_HOT_COMPILE_OPTIONS} -fno-finite-math-only)
if(NOT "${NSS_FINITE_HOT_COMPILE_OPTIONS}" STREQUAL "${_nss_finite_hot_expected}")
  message(FATAL_ERROR "NSS_FINITE_HOT_COMPILE_OPTIONS must be exactly NSS_HOT_COMPILE_OPTIONS plus -fno-finite-math-only")
endif()
if(VapourSynth_FOUND)
  file(GLOB NSS_HOST_FILTERS CONFIGURE_DEPENDS src/host/filter_*.cpp)
  add_library(nss SHARED
    src/host/plugin.cpp
    src/host/validate.cpp
    src/host/full_image.cpp
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
