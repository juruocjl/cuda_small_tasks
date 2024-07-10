# cuda_small_tasks
2024腾讯星火计划挑战周预习 - GPU高性能研发。

运行环境：R5 7500F + RTX 4070。

## Task1

[reduce_sum.cu](reduce_sum/reduce_sum.cu)

```
reduce_sum_cpu use 173.000000 ms
reduce_sum use 10.000000 ms
Check result success!
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
sort_cpu use 1744.000000 ms
bitonic_sort use 170.000000 ms
Check result success!
radix_sort_cpu use 744.000000 ms
Check result success!
radix_sort use 82.000000 ms
Check result success!
radix_sort_cub use 7.000000 ms
Check result success!
```

## Task4

[debubble.cu](debubble/debubble.cu)

```
debubble_cpu use 719.000000 ms
debubble use 64.000000 ms
Check result success
```