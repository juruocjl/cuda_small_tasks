#include "dinner123.h"
#include <cstdio>
#include <cuda_runtime.h>

__global__ void hello_world(){
  printf("Hello from %d %d\n", blockIdx.x, threadIdx.x);
}

signed main(){
  hello_world<<<5, 1024>>>();
  cudaDeviceSynchronize();
}