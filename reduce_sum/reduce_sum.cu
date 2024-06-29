#include "dinner123.h"
#include <cstring>
#include <cstdio>
#include <random>
#include <cuda_runtime.h>

__global__  void reduce_sum_kernel(float* input, size_t n, size_t dim, float* output) {
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
  Timer t;
  reduce_sum_kernel<<<grid, block>>>(i_d, n, dim, o_d);
  cudaDeviceSynchronize();
  CHECK(cudaGetLastError());
  printf("reduce_sum_gpu %ld ms\n",t.elapsed());
  CHECK(cudaMemcpy(o_h, o_d, dim * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(i_d));
  CHECK(cudaFree(o_d));
}

void reduce_sum_cpu(const float* input, size_t n, size_t dim, float* output) {
  memset(output, 0, sizeof(float) * dim);
  
  Timer t;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < dim; j+=4) {
      output[j    ] += input[i * dim + j    ];
      output[j + 1] += input[i * dim + j + 1];
      output[j + 2] += input[i * dim + j + 2];
      output[j + 3] += input[i * dim + j + 3];
    }
  }
  printf("reduce_sum_cpu %ld ms\n",t.elapsed());
}


signed main(){
  int n = 1<<15, dim = 1<<15;
  float* input_vec = (float*) malloc(n * dim * sizeof(float));
  float* output_vec1 = (float*) malloc(dim * sizeof(float));
  float* output_vec2 = (float*) malloc(dim * sizeof(float));
  initialData(input_vec, n * dim);
  reduce_sum(input_vec, n, dim, output_vec1);
  reduce_sum_cpu(input_vec, n, dim, output_vec2);
  checkResult(output_vec1, output_vec2, dim);
  free(input_vec);
  free(output_vec1);
  free(output_vec2);
}