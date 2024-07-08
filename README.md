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
mat_mul_cub kernel avg time 54.87562 ms after 100 tests
mat_mul_v4 kernel avg time 88.44581 ms after 100 tests
Check result success!
mat_mul_v5 kernel avg time 86.84523 ms after 100 tests
Check result success!
mat_mul_v6 kernel avg time 87.58622 ms after 100 tests
Check result success!
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