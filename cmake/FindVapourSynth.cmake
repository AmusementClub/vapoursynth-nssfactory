# Locate VapourSynth API4 headers (VapourSynth4.h, VSHelper4.h).
# The plugin is loaded by the host; it does not link libvapoursynth.

find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
  pkg_check_modules(PC_VapourSynth QUIET vapoursynth)
endif()

find_path(VapourSynth_INCLUDE_DIR
  NAMES VapourSynth4.h
  HINTS
    ${PC_VapourSynth_INCLUDEDIR}
    ${PC_VapourSynth_INCLUDE_DIRS}
    ENV VapourSynth_INCLUDE_DIR
  PATHS
    /usr/local/include
    /usr/include
    /usr/include/vapoursynth
    /home/owen/vapoursynth/lib/python3.14/site-packages/vapoursynth/include
  PATH_SUFFIXES vapoursynth
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(VapourSynth
  REQUIRED_VARS VapourSynth_INCLUDE_DIR
)

if(VapourSynth_FOUND AND NOT TARGET VapourSynth::VapourSynth)
  add_library(VapourSynth::VapourSynth INTERFACE IMPORTED)
  target_include_directories(VapourSynth::VapourSynth INTERFACE "${VapourSynth_INCLUDE_DIR}")
endif()

mark_as_advanced(VapourSynth_INCLUDE_DIR)
