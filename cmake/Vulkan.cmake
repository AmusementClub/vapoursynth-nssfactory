# SPDX-License-Identifier: GPL-2.0-only
# Vulkan backend dependencies (docs/vulkan-plan.md D24, D25). Every repository
# is pinned to one Vulkan SDK release so the headers, the loader shim and the
# shader tools agree. Do not float to main.
#
# Linked into libnss_vk: Vulkan-Headers (Apache-2.0 OR MIT), volk (MIT), VMA
# (MIT); see NOTICE. Build tools only, never linked: glslang and SPIRV-Tools
# (+ SPIRV-Headers). NSS_VK_GLSLANG / NSS_VK_SPIRV_OPT / NSS_VK_SPIRV_VAL may
# name installed tools instead (R5); the embedded SPIR-V then depends on their
# versions, which Backend() reports as "system". They must accept the pinned
# tools' options: glslang --depfile, spirv-opt --target-env=vulkan1.3.
set(NSS_VK_SDK_TAG "vulkan-sdk-1.4.363.0")
set(NSS_VK_VMA_TAG "v3.4.0")

set(NSS_VK_GLSLANG "" CACHE FILEPATH "Installed glslang (glslangValidator) to use instead of the pinned build")
set(NSS_VK_SPIRV_OPT "" CACHE FILEPATH "Installed spirv-opt to use instead of the pinned build")
set(NSS_VK_SPIRV_VAL "" CACHE FILEPATH "Installed spirv-val to use instead of the pinned build")

include(FetchContent)

# Header-only or self-built pieces: SOURCE_SUBDIR names a directory without a
# CMakeLists.txt, so MakeAvailable only downloads them.
FetchContent_Declare(nss_vulkan_headers
  GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
  GIT_TAG ${NSS_VK_SDK_TAG}
  GIT_SHALLOW TRUE
  SOURCE_SUBDIR nss-no-cmake)
FetchContent_Declare(nss_volk
  GIT_REPOSITORY https://github.com/zeux/volk.git
  GIT_TAG ${NSS_VK_SDK_TAG}
  GIT_SHALLOW TRUE
  SOURCE_SUBDIR nss-no-cmake)
FetchContent_Declare(nss_vma
  GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
  GIT_TAG ${NSS_VK_VMA_TAG}
  GIT_SHALLOW TRUE
  SOURCE_SUBDIR nss-no-cmake)
FetchContent_MakeAvailable(nss_vulkan_headers nss_volk nss_vma)

add_library(nss_vk_headers INTERFACE)
target_include_directories(nss_vk_headers SYSTEM INTERFACE
  "${nss_vulkan_headers_SOURCE_DIR}/include"
  "${nss_volk_SOURCE_DIR}"
  "${nss_vma_SOURCE_DIR}/include")
# volk declares every entry point; nothing may link the loader directly (D25).
target_compile_definitions(nss_vk_headers INTERFACE VK_NO_PROTOTYPES)

# volk.c is the only C source of the project (C is enabled in the root
# CMakeLists.txt, the directory common to every user of nss_volk).
add_library(nss_volk STATIC "${nss_volk_SOURCE_DIR}/volk.c")
target_link_libraries(nss_volk PUBLIC nss_vk_headers)
set_target_properties(nss_volk PROPERTIES POSITION_INDEPENDENT_CODE ON C_VISIBILITY_PRESET hidden)
if(UNIX)
  target_link_libraries(nss_volk PRIVATE ${CMAKE_DL_LIBS})
endif()

if(NSS_VK_GLSLANG AND NSS_VK_SPIRV_OPT AND NSS_VK_SPIRV_VAL)
  set(NSS_VK_GLSLANG_COMMAND "${NSS_VK_GLSLANG}")
  set(NSS_VK_SPIRV_OPT_COMMAND "${NSS_VK_SPIRV_OPT}")
  set(NSS_VK_SPIRV_VAL_COMMAND "${NSS_VK_SPIRV_VAL}")
  set(NSS_VK_SHADER_TOOLS "system")
  set(NSS_VK_GLSLANG_DEPENDS "${NSS_VK_GLSLANG}")
  set(NSS_VK_SPIRV_TOOLS_DEPENDS "${NSS_VK_SPIRV_OPT}" "${NSS_VK_SPIRV_VAL}")
elseif(NSS_VK_GLSLANG OR NSS_VK_SPIRV_OPT OR NSS_VK_SPIRV_VAL)
  message(FATAL_ERROR "NSS_VK_GLSLANG, NSS_VK_SPIRV_OPT and NSS_VK_SPIRV_VAL are set together or not at all")
else()
  FetchContent_Declare(nss_spirv_headers
    GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Headers.git
    GIT_TAG ${NSS_VK_SDK_TAG}
    GIT_SHALLOW TRUE
    SOURCE_SUBDIR nss-no-cmake)
  FetchContent_Declare(nss_spirv_tools
    GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Tools.git
    GIT_TAG ${NSS_VK_SDK_TAG}
    GIT_SHALLOW TRUE)
  FetchContent_Declare(nss_glslang
    GIT_REPOSITORY https://github.com/KhronosGroup/glslang.git
    GIT_TAG ${NSS_VK_SDK_TAG}
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(nss_spirv_headers)
  set(SPIRV-Headers_SOURCE_DIR "${nss_spirv_headers_SOURCE_DIR}" CACHE PATH "" FORCE)
  set(SPIRV_SKIP_TESTS ON CACHE BOOL "" FORCE)
  set(SPIRV_SKIP_EXECUTABLES OFF CACHE BOOL "" FORCE)
  set(SPIRV_WERROR OFF CACHE BOOL "" FORCE)
  set(SPIRV_BUILD_FUZZER OFF CACHE BOOL "" FORCE)
  set(SPIRV_TOOLS_BUILD_STATIC ON CACHE BOOL "" FORCE)
  set(SKIP_SPIRV_TOOLS_INSTALL ON CACHE BOOL "" FORCE)
  set(ENABLE_SPIRV_TOOLS_INSTALL OFF CACHE BOOL "" FORCE)
  # glslang's own optimizer is not used: every shader goes through spirv-opt.
  set(ENABLE_OPT OFF CACHE BOOL "" FORCE)
  set(ENABLE_HLSL OFF CACHE BOOL "" FORCE)
  set(ENABLE_SPVREMAPPER OFF CACHE BOOL "" FORCE)
  set(ENABLE_GLSLANG_BINARIES ON CACHE BOOL "" FORCE)
  set(GLSLANG_TESTS OFF CACHE BOOL "" FORCE)
  set(GLSLANG_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
  set(BUILD_EXTERNAL OFF CACHE BOOL "" FORCE)
  # Third-party build tools keep their own visibility and warnings.
  set(_nss_vk_saved_visibility "${CMAKE_CXX_VISIBILITY_PRESET}")
  set(CMAKE_CXX_VISIBILITY_PRESET default)
  FetchContent_MakeAvailable(nss_spirv_tools nss_glslang)
  set(CMAKE_CXX_VISIBILITY_PRESET "${_nss_vk_saved_visibility}")
  # Only the three tools the shader build depends on are built; the rest of
  # SPIRV-Tools and glslang (other tools, shared libraries) stays out of `all`.
  set_property(DIRECTORY "${nss_spirv_tools_SOURCE_DIR}" PROPERTY EXCLUDE_FROM_ALL ON)
  set_property(DIRECTORY "${nss_glslang_SOURCE_DIR}" PROPERTY EXCLUDE_FROM_ALL ON)
  foreach(_nss_tool glslang-standalone spirv-opt spirv-val)
    if(NOT TARGET ${_nss_tool})
      message(FATAL_ERROR "Pinned shader toolchain did not define target ${_nss_tool}")
    endif()
  endforeach()
  set(NSS_VK_GLSLANG_COMMAND "$<TARGET_FILE:glslang-standalone>")
  set(NSS_VK_SPIRV_OPT_COMMAND "$<TARGET_FILE:spirv-opt>")
  set(NSS_VK_SPIRV_VAL_COMMAND "$<TARGET_FILE:spirv-val>")
  set(NSS_VK_SHADER_TOOLS "glslang ${NSS_VK_SDK_TAG}, SPIRV-Tools ${NSS_VK_SDK_TAG}")
  set(NSS_VK_GLSLANG_DEPENDS glslang-standalone)
  set(NSS_VK_SPIRV_TOOLS_DEPENDS spirv-opt spirv-val)
endif()
message(STATUS "NSSFactory Vulkan: headers/volk ${NSS_VK_SDK_TAG}, VMA ${NSS_VK_VMA_TAG}, shader tools: ${NSS_VK_SHADER_TOOLS}")

# The passes of spirv-opt --loop-unroll -O (SPIRV-Tools vulkan-sdk-1.4.363.0;
# a system spirv-opt may differ) with -O's first redundancy-elimination, the
# one after loop-unroll, block-local: on large
# unrolled kernels (BM3D's warp kernel, 128-512 samples a lane) the global
# value numbering there took 6.7-25 GB and most of the time (the dev host's
# -j32 build ran out of memory); the local pass takes 0.15-1.4 GB.
set_property(GLOBAL APPEND PROPERTY JOB_POOLS nss_vk_light=8)
set(NSS_VK_SPIRV_OPT_LIGHT
  --loop-unroll --wrap-opkill --eliminate-dead-branches --merge-return --inline-entry-points-exhaustive --eliminate-dead-functions
  --eliminate-dead-code-aggressive --private-to-local --eliminate-local-single-block --eliminate-local-single-store
  --eliminate-dead-code-aggressive --scalar-replacement=0 --convert-local-access-chains --eliminate-local-single-block
  --eliminate-local-single-store --eliminate-dead-code-aggressive --ssa-rewrite --eliminate-dead-code-aggressive --ccp
  --eliminate-dead-code-aggressive --loop-unroll --eliminate-dead-branches --local-redundancy-elimination
  --combine-access-chains --simplify-instructions --scalar-replacement=0 --convert-local-access-chains
  --eliminate-local-single-block --eliminate-local-single-store --eliminate-dead-code-aggressive --ssa-rewrite
  --eliminate-dead-code-aggressive --vector-dce --eliminate-dead-inserts --eliminate-dead-branches
  --simplify-instructions --if-conversion --copy-propagate-arrays --reduce-load-size --eliminate-dead-code-aggressive
  --merge-blocks --redundancy-elimination --eliminate-dead-branches --merge-blocks --simplify-instructions)

# nss_vk_add_shader(<list-var> <id> <source.comp> [LIGHT] [DEFINES NAME=VALUE ...])
# One SPIR-V variant: glslang (GLSL 4.60 -> SPIR-V 1.6) -> spirv-opt
# --loop-unroll -O (D24 as amended 2026-10-24; LIGHT: the same passes with
# NSS_VK_SPIRV_OPT_LIGHT's local first redundancy elimination) -> spirv-val ->
# a generated .cpp holding the words. Appends the .cpp to <list-var> and the
# id to NSS_VK_SHADER_IDS / the source to NSS_VK_SHADER_SOURCES (parent scope).
function(nss_vk_add_shader out_var id source)
  cmake_parse_arguments(PARSE_ARGV 3 arg "LIGHT" "" "DEFINES")
  set(pool "")
  if(arg_LIGHT)
    set(passes ${NSS_VK_SPIRV_OPT_LIGHT})
    # Up to 1.4 GB each: at most 8 at once (Ninja), whatever -j says.
    set(pool JOB_POOL nss_vk_light)
  else()
    set(passes --loop-unroll -O)
  endif()
  if(NOT id MATCHES "^[a-z][a-z0-9_]*$")
    message(FATAL_ERROR "Shader id '${id}' must be lower_snake_case")
  endif()
  if(id IN_LIST NSS_VK_SHADER_IDS)
    message(FATAL_ERROR "Shader id '${id}' listed twice")
  endif()
  set(src "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${source}")
  if(NOT EXISTS "${src}")
    message(FATAL_ERROR "Shader variant ${id}: no source shaders/${source}")
  endif()
  set(dir "${CMAKE_CURRENT_BINARY_DIR}/spirv")
  set(raw "${dir}/${id}.raw.spv")
  set(spv "${dir}/${id}.spv")
  set(cpp "${dir}/${id}.spv.cpp")
  set(defines "")
  foreach(d IN LISTS arg_DEFINES)
    list(APPEND defines "-D${d}")
  endforeach()
  add_custom_command(
    OUTPUT "${raw}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${dir}"
    COMMAND ${NSS_VK_GLSLANG_COMMAND} -V --target-env vulkan1.3 -S comp
            "-I${CMAKE_CURRENT_SOURCE_DIR}/shaders" "-I${CMAKE_CURRENT_BINARY_DIR}/shaders_gen" ${defines}
            --depfile "${raw}.d" -o "${raw}" "${src}"
    DEPENDS "${src}" ${NSS_VK_GLSLANG_DEPENDS}
    DEPFILE "${raw}.d"
    COMMENT "glslang ${id}"
    VERBATIM)
  add_custom_command(
    OUTPUT "${spv}" "${cpp}"
    COMMAND ${NSS_VK_SPIRV_OPT_COMMAND} --target-env=vulkan1.3 ${passes} "${raw}" -o "${spv}"
    COMMAND ${NSS_VK_SPIRV_VAL_COMMAND} --target-env vulkan1.3 "${spv}"
    COMMAND ${CMAKE_COMMAND} -DNSS_VK_ID=${id} "-DNSS_VK_SPV=${spv}" "-DNSS_VK_CPP=${cpp}"
            -P "${PROJECT_SOURCE_DIR}/cmake/VulkanEmbed.cmake"
    DEPENDS "${raw}" "${PROJECT_SOURCE_DIR}/cmake/VulkanEmbed.cmake" ${NSS_VK_SPIRV_TOOLS_DEPENDS}
    ${pool}
    COMMENT "spirv-opt ${id}"
    VERBATIM)
  set(${out_var} ${${out_var}} "${cpp}" PARENT_SCOPE)
  set(NSS_VK_SHADER_IDS ${NSS_VK_SHADER_IDS} ${id} PARENT_SCOPE)
  set(NSS_VK_SHADER_SOURCES ${NSS_VK_SHADER_SOURCES} ${source} PARENT_SCOPE)
endfunction()
