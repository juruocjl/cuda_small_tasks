#include "dinner123.h"
#include <cstring>
#include <cstdio>
#include <random>
#include <cuda_runtime.h>

__global__  void vector_add_kernel(float* a, float* b, float *c){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  c[x] = a[x] + b[x];
}

void vector_add (const float* a_h, const float* b_h, float* c_h, int n) {
  float *a_d, *b_d, *c_d;
  CHECK(cudaMalloc(&a_d, n * sizeof(float)));
  CHECK(cudaMalloc(&b_d, n * sizeof(float)));
  CHECK(cudaMalloc(&c_d, n * sizeof(float)));
  CHECK(cudaMemcpy(a_d, a_h, n * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(b_d, b_h, n * sizeof(float), cudaMemcpyHostToDevice));
  dim3 block(1024);
  dim3 grid(n / block.x);
  cudaTimer t;
  vector_add_kernel<<<grid, block>>>(a_d, b_d, c_d);
  PrintTime();
  CHECK(cudaMemcpy(c_h, c_d, n * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(a_d));
  CHECK(cudaFree(b_d));
  CHECK(cudaFree(c_d));
}

void vector_add_cpu (const float* a, const float* b, float* c, int n) {
  cudaTimer t;
  for (int i = 0; i < n; i++) {
    c[i] = a[i] + b[i];
  }
  PrintTime();
}

std::mt19937 rnd(std::random_device{}());

signed main(){
  int n = 1<<25;
  float* a = (float*) malloc(n * sizeof(float));
  float* b = (float*) malloc(n * sizeof(float));
  float* out1 = (float*) malloc(n * sizeof(float));
  float* out2 = (float*) malloc(n * sizeof(float));
  for(int i = 0; i < n; i++) a[i] = rnd() & 0xff, b[i]= rnd() & 0xff;
  vector_add(a, b, out1, n);
  vector_add_cpu(a, b, out2, n);
  checkResult(out1, out2, n);
  free(a);
  free(b);
  free(out1);
  free(out2);
}