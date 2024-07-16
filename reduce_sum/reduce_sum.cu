#include "dinner123.h"
#include <cstring>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <cuda_runtime.h>

#define RUN_TIMES 100

__global__  void reduce_sum_kernel(const float* input, size_t n, size_t dim, float* output) {
  int pos=blockIdx.x * blockDim.x + threadIdx.x;
  float sum = 0;
  for (int i = 0; i < n; i++) {
    sum = sum + input[pos + i * dim];
  }
  output[pos] = sum;
}

void reduce_sum(const float* i_h, size_t n, size_t dim, float* o_h) {
  float *i_d, *o_d;
  CHECK(cudaMalloc(&i_d, n * dim * sizeof(float)));
  CHECK(cudaMalloc(&o_d, dim * sizeof(float)));
  CHECK(cudaMemcpy(i_d, i_h, n * dim * sizeof(float), cudaMemcpyHostToDevice));
  dim3 block(64);
  dim3 grid(dim / block.x);
  cudaTimer t;
  reduce_sum_kernel<<<grid, block>>>(i_d, n, dim, o_d);
  PrintTime();
  CHECK(cudaMemcpy(o_h, o_d, dim * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(i_d));
  CHECK(cudaFree(o_d));
}

void reduce_sum_cpu(const float* input, size_t n, size_t dim, float* output) {
  memset(output, 0, sizeof(float) * dim);
  
  cudaTimer t;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < dim; j++) {
      output[j] += input[i * dim + j];
    }
  }
  PrintTime();
}

#define TILE_SIZE 8

__inline__ __device__ float branch_warp_reduce_sum(float value){
  unsigned mask = __activemask();
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(mask, value, offset);
  }
  return value;
}

__global__ void reduce_sum_v2_dim1_kernel(const float* input, size_t n, size_t dim, float* output) {
  __shared__ float sum;
  if (threadIdx.x == 0) {
    sum = 0;
  }
  __syncthreads();
  float local_sum = 0;
  LIMITED_KERNEL_LOOP(i, n) {
    float d = input[i];
    local_sum += branch_warp_reduce_sum(d);
  }
  if (threadIdx.x % warpSize == 0) {
    atomicAdd(&sum, local_sum);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    atomicAdd(output, sum);
  }
}

__global__ void reduce_sum_v2_vec_kernel(const float* input, size_t n, size_t dim, float* output) {
  size_t N = n * dim;
  float local_sum = 0;
  __shared__ float sum[TILE_SIZE];
  if(threadIdx.x < TILE_SIZE) {
    sum[threadIdx.x] = 0;
  }
  __syncthreads();
  LIMITED_KERNEL_LOOP(i, N) {
    float d = input[i];
    local_sum += d;
  }
  atomicAdd(sum + threadIdx.x % TILE_SIZE, local_sum);
  __syncthreads();
  if(threadIdx.x < TILE_SIZE && dim <= threadIdx.x) {
    atomicAdd(sum + threadIdx.x % dim, sum[threadIdx.x]);
  }
  __syncthreads();
  if(threadIdx.x < dim)
    atomicAdd(output + threadIdx.x, sum[threadIdx.x]);
}

void reduce_sum_v2(const float* i_h, size_t n, size_t dim, float* o_h) {
  float *i_d, *o_d;
  CHECK(cudaMalloc(&i_d, n * dim * sizeof(float)));
  CHECK(cudaMalloc(&o_d, dim * sizeof(float)));
  CHECK(cudaMemcpy(i_d, i_h, n * dim * sizeof(float), cudaMemcpyHostToDevice));
  if (dim == 1) {
    dim3 block(1024);
    dim3 grid(multiProcessorCount);
    RUN_kernel_clear(reduce_sum_v2_dim1_kernel, grid, block, CHECK(cudaMemset(o_d, 0, dim * sizeof(float))), i_d, n, dim, o_d);
  }else if(dim <= TILE_SIZE) {
    if (TILE_SIZE % dim) throw std :: invalid_argument("Not impl");
    dim3 block(1024);
    dim3 grid(multiProcessorCount);
    RUN_kernel_clear(reduce_sum_v2_vec_kernel, grid, block, CHECK(cudaMemset(o_d, 0, dim * sizeof(float))), i_d, n, dim, o_d);
  }else throw std :: invalid_argument("Not impl");
  CHECK(cudaMemcpy(o_h, o_d, dim * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(i_d));
  CHECK(cudaFree(o_d));
}

#undef TILE_SIZE

signed main(){
  int n, dim;
  float* input_vec = (float*) malloc((1 << 30) * sizeof(float));
  initialData(input_vec, n * dim);
  {
    printf("n = %d, dim = %d\n", n = 1 << 30, dim = 1);
    float* output_vec1 = (float*) malloc(dim * sizeof(float));
    float* output_vec2 = (float*) malloc(dim * sizeof(float));
    reduce_sum_cpu(input_vec, n, dim, output_vec1);
    reduce_sum_v2(input_vec, n, dim, output_vec2);
    // checkResult(output_vec1, output_vec2, dim);
    free(output_vec1);
    free(output_vec2);
  }
  {
    printf("n = %d, dim = %d\n", n = 1 << 29, dim = 2);
    float* output_vec1 = (float*) malloc(dim * sizeof(float));
    float* output_vec2 = (float*) malloc(dim * sizeof(float));
    reduce_sum_cpu(input_vec, n, dim, output_vec1);
    reduce_sum_v2(input_vec, n, dim, output_vec2);
    // checkResult(output_vec1, output_vec2, dim);
    free(output_vec1);
    free(output_vec2);
  }
  {
    printf("n = %d, dim = %d\n", n = 1 << 28, dim = 4);
    float* output_vec1 = (float*) malloc(dim * sizeof(float));
    float* output_vec2 = (float*) malloc(dim * sizeof(float));
    reduce_sum_cpu(input_vec, n, dim, output_vec1);
    reduce_sum_v2(input_vec, n, dim, output_vec2);
    // checkResult(output_vec1, output_vec2, dim);
    free(output_vec1);
    free(output_vec2);
  }
  {
    printf("n = %d, dim = %d\n", n = 1 << 27, dim = 8);
    float* output_vec1 = (float*) malloc(dim * sizeof(float));
    float* output_vec2 = (float*) malloc(dim * sizeof(float));
    reduce_sum_cpu(input_vec, n, dim, output_vec1);
    reduce_sum_v2(input_vec, n, dim, output_vec2);
    // checkResult(output_vec1, output_vec2, dim);
    free(output_vec1);
    free(output_vec2);
  }
  free(input_vec);
}