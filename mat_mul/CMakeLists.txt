add_executable(mat_mul mat_mul.cu)
target_link_libraries(mat_mul PRIVATE cublas)