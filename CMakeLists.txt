cmake_minimum_required(VERSION 3.14)
project(cuda_matrix C CUDA)

find_package(CUDA 10.0 REQUIRED)

set(CMAKE_C_STANDARD 99)

if (WIN32)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
    set(BUILD_SHARED_LIBS TRUE)
endif()

add_library(matrix SHARED src/lib/matrix.cu)
set_target_properties(matrix PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


##### Tests #####
add_executable(cuda_env_test test/cuda_env_test.cu)

add_executable(matrix_test test/matrix_test.c)
set_property(TARGET matrix_test PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(matrix_test matrix)
target_include_directories(matrix_test PUBLIC src/lib)
