# cuda_small_tasks
2024腾讯星火计划挑战周预习 - GPU高性能研发。

运行环境：R5 7500F + RTX 4070。

## Task1

[reduce_sum.cu](reduce_sum/reduce_sum.cu)

```
n = 1073741824, dim = 1
reduce_sum_cpu use 2715.870117 ms
reduce_sum_v2_dim1_kernel avg use 10.94559 ms in 100 tests
n = 536870912, dim = 2
reduce_sum_cpu use 1306.851318 ms
reduce_sum_v2_vec_kernel avg use 9.52499 ms in 100 tests
n = 268435456, dim = 4
reduce_sum_cpu use 666.475952 ms
reduce_sum_v2_vec_kernel avg use 9.52111 ms in 100 tests
n = 134217728, dim = 8
reduce_sum_cpu use 326.891998 ms
reduce_sum_v2_vec_kernel avg use 9.43696 ms in 100 tests
```

## Task2

[mat_mul.cu](mat_mul/mat_mul.cu)
```
cublasSgemm avg use 7.14007 ms in 100 tests
mat_mul_v1_kernel avg use 589.63213 ms in 100 tests
mat_mul_v2_kernel avg use 144.05709 ms in 100 tests
mat_mul_v3_kernel avg use 29.23753 ms in 100 tests
mat_mul_v4_kernel avg use 10.12860 ms in 100 tests
mat_mul_v5_kernel avg use 10.69626 ms in 100 tests
mat_mul_v6_kernel avg use 10.45179 ms in 100 tests
mat_mul_v7_kernel avg use 7.30132 ms in 100 tests
mat_mul_v8_kernel avg use 7.34459 ms in 100 tests
```

## Task3

[sort.cu](sort/sort.cu)

```
radix_sort_cub use 22.262913 ms
radix_sort use 630.698608 ms
Check result success!
radix_sort_v2 use 306.193298 ms
Check result success!
```

## Task4

[debubble.cu](debubble/debubble.cu)

```
debubble_cpu use 706.000000 ms
debubble use 62.070305 ms
Check result success!
debubble_v2 use 7.996352 ms
Check result success!
```