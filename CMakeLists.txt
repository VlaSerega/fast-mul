cmake_minimum_required(VERSION 3.21)
project(FastMulHuaweiTask C)

set(CMAKE_C_STANDARD 23)

add_compile_options(-O3 -mavx -mfma)

set(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -fopenmp")

add_executable(FastMulHuaweiTask main.c fast_mul.c fast_mul.h)

target_link_libraries(FastMulHuaweiTask -L/home/vlaserega/CBLAS/lib)