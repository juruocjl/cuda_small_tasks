#include "dinner123.h"
#include <cstring>
#include <cstdio>
#include <cassert>
#include <random>
#include <cuda_runtime.h>

size_t debubble_cpu(vector<int> &a) {
  Timer t;
  size_t pos = 0, n = a.size();
  for (int i = 0; i < a.size(); i++) {
    if (a[i]) {
      a[pos++] = a[i];
    }
  }
  for (int i = pos; i < n; i++) {
    a[i] = 0;
  }
  PrintTime();
  return pos;
}

mt19937 rnd(random_device{}());

size_t debubble_tmp(vector<int> &a) {
  Timer t;
  size_t n = a.size();
  int *sum, *b;
  sum = (int*)malloc(n * sizeof(int));
  b = (int*)malloc(n * sizeof(int));
  for (int i = 0; i < n; i++) {
    sum[i] = a[i] != 0;
  }
  for (int i = 1; i < n; i *= 2) {
    for (int j = i - 1 ; j + i < n; j += i * 2) {
      sum[i + j] += sum[j];
    }
  }
  for (int i = n / 4; i; i /= 2) {
    for (int j = i * 2 - 1; j + i < n; j += i * 2) {
      sum [i + j] += sum[j];
    }
  }
  for (int i = 0; i < n; i++) {
    if (a[i]) {
      b[sum[i] - 1] = a[i];
    }
  }
  int res = sum[n - 1];
  memcpy(a.data(), b, n * sizeof(int));
  free(b);
  free(sum);
  PrintTime();
  return res;
}
__global__ void sum1_kernel(int*sum, int i){
  int j = i - 1 + i * 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  sum[i + j] += sum[j];
}
__global__ void sum2_kernel(int*sum, int i){
  if (blockIdx.x + 1 != gridDim.x || threadIdx.x + 1 != blockDim.x) {
    int j = i * 2 - 1 + i * 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    sum[i + j] += sum[j];
  }
}
void presum(int *a, int n){
  for (int i = 1; i < n; i *= 2){
    int tot = n / i / 2;
    if(tot < 1024) sum1_kernel<<<1, tot>>>(a, i);
    else sum1_kernel<<<tot / 1024, 1024>>>(a, i);
  }
  for (int i = n / 4; i; i /= 2) {
    int tot = n / i / 2;
    if(tot < 1024) sum2_kernel<<<1, tot>>>(a, i);
    else sum2_kernel<<<tot / 1024, 1024>>>(a, i);
  }
}
__global__ void debubble_pre_kernel(int *a, int *sum){
  int pos = blockIdx.x * blockDim.x + threadIdx.x;
  sum[pos] = a[pos] != 0;
}
__global__ void debubble_place_kernel(int *a, int *sum,int *b){
  int pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (a[pos]) {
    b[sum[pos] - 1] = a[pos];
  }
}
size_t debubble(vector<int> &a) {
  size_t n = a.size();
  size_t Bytes = n * sizeof(int);
  int res;
  int *a_d, *s_d, *b_d;
  CHECK(cudaMalloc(&a_d, Bytes));
  CHECK(cudaMalloc(&s_d, Bytes));
  CHECK(cudaMalloc(&b_d, Bytes));
  CHECK(cudaMemcpy(a_d, a.data(), Bytes, cudaMemcpyHostToDevice));
  cudaTimer t;
  debubble_pre_kernel<<<n / 1024, 1024>>>(a_d, s_d);
  presum(s_d, n);
  debubble_place_kernel<<<n / 1024, 1024>>>(a_d, s_d, b_d);
  PrintTime();
  CHECK(cudaMemcpy(a.data(), b_d, Bytes, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(&res, s_d + (n - 1), sizeof(int), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(a_d));
  CHECK(cudaFree(s_d));
  CHECK(cudaFree(b_d));
  return res;
}
#define TILE_SIZE 32
__global__ void debubble_v2_pre_kernel(int *a, int *s, int n){
  cg :: thread_block_tile<TILE_SIZE> g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  int rank = g.thread_rank();
  LIMITED_KERNEL_LOOP(i, n) {
    bool is_existed = a[i] != 0;
    unsigned mask = g.ballot(is_existed);
    if (rank == 0) {
      s[i / TILE_SIZE] = __popc(mask);
    }
  }
}
__global__ void debubble_v2_place_kernel(int *a, int *s, int *b, int n){
  cg :: thread_block_tile<TILE_SIZE> g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  int rank = g.thread_rank();
  LIMITED_KERNEL_LOOP(i, n) {
    bool is_existed = a[i] != 0;
    unsigned mask = g.ballot(is_existed);
    if (is_existed) {
      int bias = s[i / TILE_SIZE] - __popc(mask >> rank);
      b[bias] = a[i];
    }
  }
}

size_t debubble_v2(vector<int> &a) {
  size_t n = a.size();
  size_t Bytes = n * sizeof(int);
  int res;
  int *a_d, *s_d, *b_d;
  CHECK(cudaMalloc(&a_d, Bytes));
  CHECK(cudaMalloc(&s_d, Bytes / TILE_SIZE + sizeof(int)));
  CHECK(cudaMalloc(&b_d, Bytes));
  CHECK(cudaMemcpy(a_d, a.data(), Bytes, cudaMemcpyHostToDevice));
  cudaTimer t;
  debubble_v2_pre_kernel<<<multiProcessorCount, 1024>>>(a_d, s_d, n);
  presum(s_d, n / TILE_SIZE);
  debubble_v2_place_kernel<<<multiProcessorCount, 1024>>>(a_d, s_d, b_d, n);
  PrintTime();
  CHECK(cudaMemcpy(a.data(), b_d, Bytes, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(&res, s_d + (n / TILE_SIZE - 1), sizeof(int), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(a_d));
  CHECK(cudaFree(s_d));
  CHECK(cudaFree(b_d));
  return res;
}
#undef TILE_SIZE
signed main() {
  int n = 1 << 28;
  vector<int> A(n), STD, B;
  for (int i = 0; i < n; i++) {
    if (rnd() & 1) A[i] = 0;
    else A[i] = rnd();
  }
  STD = A;
  int res = debubble_cpu(STD);
  B = A;
  assert(res == debubble(B));
  checkResult(B, STD);
  B = A;
  assert(res == debubble_v2(B));
  checkResult(B, STD);
}