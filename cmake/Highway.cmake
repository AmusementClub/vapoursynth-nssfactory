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
