cmake_minimum_required(VERSION 3.0.0)
project(small_tasks VERSION 0.1.0 LANGUAGES C CXX CUDA)

include_directories(./include)
add_subdirectory(./hello_world)
add_subdirectory(./vector_add)
add_subdirectory(./reduce_sum)