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

void radix_sort_cpu(std :: vector<int> &nums) {
  int n = nums.size();
  std :: vector<int>nums_(nums.size()),a(1<<16),b(1<<16);
  Timer t;
  for(int i = 0; i < n; i++) nums[i] = ((unsigned)nums[i]) ^  0x80000000;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < (1<<16); j++) {
      a [j] = 0;
    }
    for (int j = 0; j < n; j++) {
      a[((unsigned)nums[j]) >> (i << 4) & 0xffff]++;
    }
    b[0] = 0;
    for (int j = 1; j < (1<<16); j++) {
      b[j] = b[j - 1] + a[j - 1];
    }
    for (int j = 0; j < n; j++) {
      nums_[b[((unsigned)nums[j]) >> (i << 4) & 0xffff]++] = nums[j]; 
    }
    std :: swap(nums, nums_);
  }
  for(int i = 0; i < n; i++) nums[i] = ((unsigned)nums[i]) ^  0x80000000;
  PrintTime();
}


__global__ void radix_sort_count_kernel(uint *a1, uint *cnt, uint shift) {
  __shared__ uint tmp[256];
  if (threadIdx.x < 256) {
    tmp[threadIdx.x] = 0;
  }
  __syncthreads();
  atomicAdd(&tmp[(a1[blockIdx.x * 1024 + threadIdx.x] >> shift & 0xff)], 1);
  __syncthreads();
  if (threadIdx.x < 256) {
    cnt[blockIdx.x * 256 + threadIdx.x] = tmp[threadIdx.x];
  }
}
__global__ void radix_sort_prefix_kernel(uint *cnt, uint *sum, uint n){
  int x = threadIdx.x;
  for (uint i = 1; i < n; i++) {
    cnt[i * 256 + x] += cnt[(i - 1) * 256 + x];
  }
  sum[x] = cnt[(n - 1) * 256 + x];
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int i = 0; i < 255; i++) {
      sum[i + 1] += sum[i];
    }
  }
    
}
__global__ void radix_sort_place_kernel(uint *a1, uint *a2, uint *cnt, uint *sum, int shift){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = 1; i < 256; i++)
    cnt[x * 256 + i] += sum[i - 1];
  for (uint i = 1023; ~i; i--) {
    uint v = a1[x * 1024 + i];
    a2[--cnt[x * 256 + (v >> shift & 0xff)]] = v;
  }
}

void radix_sort(std::vector<int> &nums) {
  uint n = nums.size(), Bytes = n * sizeof(int);
  for(uint i = 0; i < n; i++) nums[i] = ((unsigned)nums[i]) ^ 0x80000000;
  uint *a1, *a2, *cnt, *sum;
  CHECK(cudaMalloc(&a1, Bytes));
  CHECK(cudaMalloc(&a2, Bytes));
  CHECK(cudaMalloc(&cnt, n / 1024 * 256 * sizeof(int)));
  CHECK(cudaMalloc(&sum, 256 * sizeof(int)));
  CHECK(cudaMemcpy(a1, nums.data(), Bytes, cudaMemcpyHostToDevice));
  Timer t;
  for (uint i = 0; i < 4; i++) {
    radix_sort_count_kernel<<<n / 1024, 1024>>>(a1, cnt, i * 8);
    radix_sort_prefix_kernel<<<1, 256>>>(cnt, sum, n / 1024);
    radix_sort_place_kernel<<<n / 1024, 1>>>(a1, a2, cnt, sum, i * 8);
    std :: swap(a1, a2);
  }
  CHECK(cudaDeviceSynchronize());
  PrintTime();
  CHECK(cudaMemcpy(nums.data(), a1, Bytes, cudaMemcpyDeviceToHost));
  for(uint i = 0; i < n; i++) nums[i] = ((unsigned)nums[i]) ^  0x80000000;
  CHECK(cudaFree(a1));
  CHECK(cudaFree(a2));
  CHECK(cudaFree(cnt));
  CHECK(cudaFree(sum));
}


signed main(){
  size_t n = 1 << 25;
  std :: mt19937 rnd(random_device{}());
  std :: vector<int> A(n),STD,B;
  for (int i = 0; i < n; i++) {
    A[i] = rnd();
  }
  STD=A;
  sort_cpu(STD);
  B = A;
  bitonic_sort_cpu(B);
  checkResult(STD,B);
  B = A;
  bitonic_sort(B);
  checkResult(STD,B);
  B = A;
  radix_sort_cpu(B);
  checkResult(STD,B);
  B = A;
  radix_sort(B);
  checkResult(STD,B);
  
}