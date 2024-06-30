#include "dinner123.h"
#include <cstring>
#include <cstdio>
#include <random>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

void sort_cpu(std :: vector<int> &nums) {
  Timer t;
  std :: sort(nums.begin(),nums.end());
  PrintTime();
}

void bitonic_sort_cpu(std :: vector<int> &nums) {
  Timer t;
  size_t n = nums.size();
  for (size_t L = 0; (1 << L) < n; L++) {
    for (size_t LL = L; ~ LL; LL--) {
      for (size_t i = 0; i < n; i += (1 << (LL + 1))) {
        for (size_t j = 0; j < (1 << LL); j++) {
          if ((nums[i + j] < nums[i + j + (1 << LL)]) == (i >> (L + 1) & 1)) {
            std :: swap(nums[i + j], nums[i + j + (1 << LL)]);
          }
        }
      }
    }
  }
  PrintTime();
}

#define BLOCK_SIZE 1024
__global__ void bitonic_sort_kernel(int *a, int L, int LL) {
  int pos = threadIdx.x + BLOCK_SIZE * blockIdx.x;
  int j = pos & ((1 << LL) - 1), i = (pos - j) << 1;
  if ((a[i + j] < a[i + j + (1 << LL)]) == (i >> (L + 1) & 1)) {
    int t = a[i + j];
    a[i + j] = a[i + j + (1 << LL)];
    a[i + j + (1 << LL)] = t;
  }
}
void bitonic_sort(std :: vector<int> &nums) {
  size_t n = nums.size();
  size_t Bytes = n * sizeof(int);
  int *a_d;
  CHECK(cudaMalloc(&a_d, Bytes));
  CHECK(cudaMemcpy(a_d, nums.data(), Bytes, cudaMemcpyHostToDevice));
  Timer t;
  for (size_t L = 0; (1 << L) < n; L++) {
    for (size_t LL = L; ~ LL; LL--) {
      bitonic_sort_kernel<<<n / 2 / BLOCK_SIZE, BLOCK_SIZE>>>(a_d, L, LL);
    }
  }
  CHECK(cudaDeviceSynchronize());
  PrintTime();
  CHECK(cudaMemcpy(nums.data(), a_d, Bytes, cudaMemcpyDeviceToHost));
  CHECK(cudaFree(a_d));
}
#undef BLOCK_SIZE

signed main(){
  size_t n = 1 << 30;
  std :: mt19937 rnd(random_device{}());
  std :: vector<int> A(n);
  for (int i = 0; i < n; i++) {
    A[i] = rnd();
  }
  auto B = A;
  sort_cpu(A);
  bitonic_sort(B);
  checkResult(A,B);
}