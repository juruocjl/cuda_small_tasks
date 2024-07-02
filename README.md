# cuda_small_tasks
2024腾讯星火计划挑战周预习 - GPU高性能研发。

运行环境：R7 4800H + RTX 2060。

## Task1

[reduce_sum.cu](reduce_sum/reduce_sum.cu)

```
reduce_sum_cpu use 290 ms
reduce_sum use 27 ms
Check result success!
```

## Task2

[mat_mul.cu](mat_mul/mat_mul.cu)
```
mat_mul_cpu use 10529 ms
mat_mul_v1 use 2332 ms
Check result success!
mat_mul_v2 use 566 ms
Check result success!
mat_mul_cub use 41 ms
Check result success!
mat_mul_v3 use 102 ms
Check result success!
```

## Task3

[sort.cu](sort/sort.cu)

```
sort_cpu use 3368 ms
bitonic_sort use 338 ms
Check result success!
radix_sort_cpu use 538 ms
Check result success!
radix_sort use 171 ms
Check result success!
radix_sort_cub use 23 ms
Check result success!
```

## Task4

[debubble.cu](debubble/debubble.cu)

```
debubble_cpu use 1375 ms
debubble use 155 ms
Check result success!
```