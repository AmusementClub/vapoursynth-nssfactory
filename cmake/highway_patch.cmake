# SPDX-License-Identifier: GPL-2.0-only
# Highway's patch step (cmake/Highway.cmake): comments out its
# cmake_policy(SET CMP0111 OLD), which only GTest imports need. Idempotent.
# usage: cmake -DFILE=<highway>/CMakeLists.txt -P highway_patch.cmake
file(READ "${FILE}" text)
string(REPLACE "  cmake_policy(SET CMP0111 OLD)" "  # cmake_policy(SET CMP0111 OLD)  (nssfactory: no GTest imports)" text "${text}")
file(WRITE "${FILE}" "${text}")
