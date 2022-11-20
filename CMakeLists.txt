cmake_minimum_required(VERSION 3.16)
project(lab7_matrix)

set(CMAKE_CXX_STANDARD 14)
add_compile_options(-I/usr/lib/x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu -O3 -funsafe-math-optimizations -march=native -lopenblas)
add_executable(lab7_matrix main.cpp)
