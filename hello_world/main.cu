#include "dinner123.h"
#include <cstdio>
#include <cuda_runtime.h>

__global__ void hello_world(){
  printf("Hello from %d\n", threadIdx.x);
}

signed main(){
  hello_world<<<1,10>>>();
  cudaDeviceSynchronize();
}