# Google Highway, pinned release tag. Do not float to master.
# Tag 1.4.0 — current stable as of 2026-04-23 (github.com/google/highway/releases).
set(NSS_HIGHWAY_GIT_TAG "1.4.0")

include(FetchContent)
FetchContent_Declare(
  highway
  GIT_REPOSITORY https://github.com/google/highway.git
  GIT_TAG ${NSS_HIGHWAY_GIT_TAG}
  GIT_SHALLOW TRUE
)
set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(highway)

# Keep the bundled diagnostic helper aligned with the project dispatch mask.
# The hwy library itself remains target-agnostic; business TUs receive the
# same definition through nss_cpu below and through hwy_config.hpp.
if(TARGET hwy_list_targets)
  target_compile_definitions(hwy_list_targets PRIVATE "HWY_DISABLED_TARGETS=${NSS_HWY_DISABLED_TARGETS}")
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64|i[3-6]86)$")
    target_compile_options(hwy_list_targets PRIVATE -mavx2 -mfma)
  endif()
endif()
