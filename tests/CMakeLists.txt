add_executable(test_svd test_svd.cpp)
target_link_libraries(test_svd PRIVATE nss_cpu)
target_include_directories(test_svd PRIVATE "${PROJECT_SOURCE_DIR}/include")

add_executable(test_dct8 test_dct8.cpp)
target_link_libraries(test_dct8 PRIVATE nss_cpu)
target_include_directories(test_dct8 PRIVATE "${PROJECT_SOURCE_DIR}/include")

add_executable(test_nlm_cpu test_nlm_cpu.cpp)
target_link_libraries(test_nlm_cpu PRIVATE nss_cpu)
target_include_directories(test_nlm_cpu PRIVATE "${PROJECT_SOURCE_DIR}/include")

add_test(NAME test_svd COMMAND test_svd)
add_test(NAME test_dct8 COMMAND test_dct8)
add_test(NAME test_nlm_cpu COMMAND test_nlm_cpu)

add_executable(test_common test_common.cpp)
target_link_libraries(test_common PRIVATE nss_cpu)
target_include_directories(test_common PRIVATE "${PROJECT_SOURCE_DIR}/include")
add_test(NAME test_common COMMAND test_common)

foreach(name mcwnnm twsc nlh ncsr lssc)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_${name}.cpp")
    add_executable(test_${name} test_${name}.cpp)
    target_link_libraries(test_${name} PRIVATE nss_cpu)
    target_include_directories(test_${name} PRIVATE "${PROJECT_SOURCE_DIR}/include")
    add_test(NAME test_${name} COMMAND test_${name})
  endif()
endforeach()

add_executable(bench_cpu bench_cpu.cpp)
target_link_libraries(bench_cpu PRIVATE nss_cpu)
target_include_directories(bench_cpu PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_compile_options(bench_cpu PRIVATE -O3)
