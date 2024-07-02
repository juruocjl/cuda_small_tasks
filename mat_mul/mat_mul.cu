#include "dinner123.h"
#include <cstring>
#include <cstdio>
#include <random>
#include <cuda_runtime.h>
#include <cublas_v2.h>

void mat_mul_cpu(const float *A, const float *B, size_t m, size_t n, size_t k, float *output) {
  Timer t;
  memset(output, 0, m * k * sizeof(float));
  for (int x = 0; x < m; x++) {
    for (int y = 0; y < n; y++) {
      for (int z = 0; z < k; z++) {
        output[x * k + z] += A[x * n + y] * B[y * k + z];
      }
    }
  }
  PrintTime();
}

__global__ void mat_mul_v1_kernel (const float *A, const float *B, size_t m, size_t n, size_t k, float *C) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  float sum = 0;
  for (int i = 0; i < n; i++) {
    sum = sum + A[x * n + i] * B[i * k + y];
  }
  C[x * k + y] = sum;
}

void mat_mul_v1(const float *A_h, const float *B_h, size_t m, size_t n, size_t k, float *C_h) {
  float *A_d, *B_d, *C_d;
  CHECK(cudaMalloc(&A_d, m * n * sizeof(float)));
  CHECK(cudaMalloc(&B_d, n * k * sizeof(float)));
  CHECK(cudaMalloc(&C_d, m * k * sizeof(float)));
  CHECK(cudaMemcpy(A_d, A_h, m * n * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(B_d, B_h, n * k * sizeof(float), cudaMemcpyHostToDevice));
  Timer t;
  dim3 block(32, 32);
  dim3 grid(m / block.x, k / block.y);
  mat_mul_v1_kernel<<<grid, block>>>(A_d, B_d, m, n, k, C_d);
  cudaDeviceSynchronize();
  CHECK(cudaGetLastError());
  PrintTime();
  CHECK(cudaMemcpy(C_h, C_d, m * k * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(A_d));
  CHECK(cudaFree(B_d));
  CHECK(cudaFree(C_d));
}

#define BLOCK_SIZE 16

__global__ void mat_mul_v2_kernel (const float *A, const float *B, size_t m, size_t n, size_t k, float *C) {
  int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;
  float sum = 0;
  for (int i = 0; i < n / BLOCK_SIZE; i++) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    As[threadIdx.x][threadIdx.y] = A[x * n + i * BLOCK_SIZE + threadIdx.y];
    Bs[threadIdx.x][threadIdx.y] = B[(i * BLOCK_SIZE + threadIdx.x) * k + y];
    __syncthreads();
    for (int j = 0; j < BLOCK_SIZE; j++) {
      sum += As[threadIdx.x][j] * Bs[j][threadIdx.y];
    }
    __syncthreads();
  }
  C[x * k + y] = sum;
}

void mat_mul_v2(const float *A_h, const float *B_h, size_t m, size_t n, size_t k, float *C_h) {
  float *A_d, *B_d, *C_d;
  CHECK(cudaMalloc(&A_d, m * n * sizeof(float)));
  CHECK(cudaMalloc(&B_d, n * k * sizeof(float)));
  CHECK(cudaMalloc(&C_d, m * k * sizeof(float)));
  CHECK(cudaMemcpy(A_d, A_h, m * n * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(B_d, B_h, n * k * sizeof(float), cudaMemcpyHostToDevice));
  Timer t;
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid(m / BLOCK_SIZE, k / BLOCK_SIZE);
  mat_mul_v2_kernel<<<grid, block>>>(A_d, B_d, m, n, k, C_d);
  cudaDeviceSynchronize();
  CHECK(cudaGetLastError());
  PrintTime();
  CHECK(cudaMemcpy(C_h, C_d, m * k * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(A_d));
  CHECK(cudaFree(B_d));
  CHECK(cudaFree(C_d));
}
#undef BLOCK_SIZE

void mat_mul_cub(const float *A_h, const float *B_h, size_t m, size_t n, size_t k, float *C_h) {
  float *A_d, *B_d, *C_d;
  CHECK(cudaMalloc(&A_d, m * n * sizeof(float)));
  CHECK(cudaMalloc(&B_d, n * k * sizeof(float)));
  CHECK(cudaMalloc(&C_d, m * k * sizeof(float)));
  CHECK(cudaMemcpy(A_d, A_h, m * n * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(B_d, B_h, n * k * sizeof(float), cudaMemcpyHostToDevice));
  Timer t;
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  const float alpha = 1.0f;
  const float beta = 0.0f;
  CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                             k, m, n, &alpha, 
                             B_d, k, A_d, n, 
                             &beta, C_d, k));
  CHECK(cudaDeviceSynchronize());
  PrintTime();
  CHECK(cudaMemcpy(C_h, C_d, m * k * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(A_d));
  CHECK(cudaFree(B_d));
  CHECK(cudaFree(C_d));
}

#define BLOCK_SIZE 8
#define MAT_SIZE 8

__global__ void mat_mul_v3_kernel (const float4 *A, const float4 *B, size_t m, size_t n, size_t k, float *C) {
  int subx = threadIdx.x * MAT_SIZE;
  int suby = threadIdx.y * MAT_SIZE;
  int x = blockIdx.x * BLOCK_SIZE * MAT_SIZE + subx;
  int y = blockIdx.y * BLOCK_SIZE * MAT_SIZE + suby;
  float sum[MAT_SIZE][MAT_SIZE];
  for (int i = 0; i < MAT_SIZE; i++) {
    for (int j = 0; j < MAT_SIZE; j++) {
      sum[i][j] = 0;
    }
  }
  for (int i = 0; i < n / BLOCK_SIZE / MAT_SIZE; i++) {
    __shared__ float4 As[BLOCK_SIZE * MAT_SIZE][BLOCK_SIZE * MAT_SIZE / 4];
    __shared__ float4 Bs[BLOCK_SIZE * MAT_SIZE][BLOCK_SIZE * MAT_SIZE / 4];
    #pragma unroll
    for (int a = 0; a < MAT_SIZE; a++) {
      #pragma unroll
      for (int b = 0; b < MAT_SIZE; b += 4) {
        As[subx + a][(suby + b) >> 2] = A[((x + a) * n + i * BLOCK_SIZE * MAT_SIZE + suby + b) >> 2];
        Bs[suby + a][(subx + b) >> 2] = B[((y + a) * n + i * BLOCK_SIZE * MAT_SIZE + subx + b) >> 2];
      }
    }
    __syncthreads();
    float4 Ass, Bss;
    for (int j = 0; j < BLOCK_SIZE; j++) {
      #pragma unroll
      for (int a = 0; a < MAT_SIZE; a++) {
        #pragma unroll
        for (int b = 0; b < MAT_SIZE; b++) {
          #pragma unroll
          float s = 0;
          for (int c = 0; c < MAT_SIZE / 4; c++) {
            Ass = As[subx + a][j * MAT_SIZE / 4 + c];
            Bss = Bs[suby + b][j * MAT_SIZE / 4 + c];
            s += Ass.x * Bss.x + Ass.y * Bss.y + Ass.z * Bss.z + Ass.w * Bss.w;
          }
          sum[a][b] += s;
        }
      }
    }
    __syncthreads();
  }
  for (int a = 0; a < MAT_SIZE; a++) {
    for (int b = 0; b < MAT_SIZE; b++) {
      C[(x + a) * k + y + b] = sum[a][b];
    }
  }
}

void mat_mul_v3(const float *A_h, const float *B_h, size_t m, size_t n, size_t k, float *C_h) {
  float *B_T_h = (float*)malloc(n * k * sizeof(float));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < k; j++) {
      B_T_h[j * n + i] = B_h[i * k + j];
    }
  }
  float4 *A_d, *B_d;
  float *C_d;
  CHECK(cudaMalloc(&A_d, m * n * sizeof(float)));
  CHECK(cudaMalloc(&B_d, n * k * sizeof(float)));
  CHECK(cudaMalloc(&C_d, m * k * sizeof(float)));
  CHECK(cudaMemcpy(B_d, B_T_h, n * k * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(A_d, A_h, m * n * sizeof(float), cudaMemcpyHostToDevice));
  Timer t;
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid(m / BLOCK_SIZE / MAT_SIZE, k / BLOCK_SIZE / MAT_SIZE);
  mat_mul_v3_kernel<<<grid, block>>>(A_d, B_d, m, n, k, C_d);
  cudaDeviceSynchronize();
  CHECK(cudaGetLastError());
  PrintTime();
  CHECK(cudaMemcpy(C_h, C_d, m * k * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(A_d));
  CHECK(cudaFree(B_d));
  CHECK(cudaFree(C_d));
  free(B_T_h);
}
#undef BLOCK_SIZE
#undef MAT_SIZE

signed main(){
  int m = 1 << 12, n = 1 << 12, k = 1 << 11;
  float *A, *B, *STD, *OUT;
  A = (float*)malloc(m * n * sizeof(float));
  B = (float*)malloc(n * k * sizeof(float));
  STD = (float*)malloc(m * k * sizeof(float));
  OUT = (float*)malloc(m * k * sizeof(float));
  
  initialData(A, m * n);
  initialData(B, n * k);
  mat_mul_cpu(A, B, m, n, k, STD);
  mat_mul_v1(A, B, m, n, k, OUT);
  checkResult(STD, OUT ,m * k);
  mat_mul_v2(A, B, m, n, k, OUT);
  checkResult(STD, OUT ,m * k);
  mat_mul_cub(A, B, m, n, k, OUT);
  checkResult(STD, OUT ,m * k);
  mat_mul_v3(A, B, m, n, k, OUT);
  checkResult(STD, OUT ,m * k);
  free(A);
  free(B);
  free(STD);
  free(OUT);
}