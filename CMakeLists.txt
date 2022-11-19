cmake_minimum_required(VERSION 3.16)
project(lab7_matrix)

set(CMAKE_CXX_STANDARD 14)
add_compile_options(-O3 -msse2 -funsafe-math-optimizations -fopt-info-vec-missed)
add_executable(lab7_matrix main.cpp)
