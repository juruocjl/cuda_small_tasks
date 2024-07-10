#include "dinner123.h"
#include <cstring>
#include <cstdio>
#include <random>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#define RUN_TIMES 100

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

void mat_mul_cub(const float *A_h, const float *B_h, size_t m, size_t n, size_t k, float *C_h) {
  float *A_d, *B_d, *C_d;
  CHECK(cudaMalloc(&A_d, m * n * sizeof(float)));
  CHECK(cudaMalloc(&B_d, n * k * sizeof(float)));
  CHECK(cudaMalloc(&C_d, m * k * sizeof(float)));
  CHECK(cudaMemcpy(A_d, A_h, m * n * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(B_d, B_h, n * k * sizeof(float), cudaMemcpyHostToDevice));
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  const float alpha = 1.0f;
  const float beta = 0.0f;
  RUN(cublasSgemm, handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                   k, m, n, &alpha, 
                   B_d, k, A_d, n, 
                   &beta, C_d, k);
  CHECK(cudaMemcpy(C_h, C_d, m * k * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(A_d));
  CHECK(cudaFree(B_d));
  CHECK(cudaFree(C_d));
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
  dim3 block(32, 32);
  dim3 grid(m / block.x, k / block.y);
  RUN_kernel(mat_mul_v1_kernel, grid, block, A_d, B_d, m, n, k, C_d);
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
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid(m / BLOCK_SIZE, k / BLOCK_SIZE);
  RUN_kernel(mat_mul_v2_kernel, grid, block, A_d, B_d, m, n, k, C_d);
  CHECK(cudaMemcpy(C_h, C_d, m * k * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(A_d));
  CHECK(cudaFree(B_d));
  CHECK(cudaFree(C_d));
}
#undef BLOCK_SIZE


#define BLOCK_SIZE 8
#define MAT_SIZE 8

__global__ void mat_mul_v3_kernel (const float4 *A, const float4 *B, size_t m, size_t n, size_t k, float *C) {
  int subx = threadIdx.x * MAT_SIZE;
  int suby = threadIdx.y * MAT_SIZE;
  int blockx = blockIdx.x * BLOCK_SIZE * MAT_SIZE;
  int blocky = blockIdx.y * BLOCK_SIZE * MAT_SIZE;
  int x = blockx + subx;
  int y = blocky + suby;
  int tid = threadIdx.x * BLOCK_SIZE + threadIdx.y;
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
    for (int b = 0; b < MAT_SIZE * MAT_SIZE; b += 4) {
      As[tid][b >> 2] = A[((blockx + tid) * n + i * BLOCK_SIZE * MAT_SIZE + b) >> 2];
      Bs[tid][b >> 2] = B[((blocky + tid) * n + i * BLOCK_SIZE * MAT_SIZE + b) >> 2];
    }
    __syncthreads();
    float4 Ass, Bss;
    for (int j = 0; j < BLOCK_SIZE; j++) {
      #pragma unroll
      for (int a = 0; a < MAT_SIZE; a++) {
        #pragma unroll
        for (int b = 0; b < MAT_SIZE; b++) {
          float s = 0;
          #pragma unroll
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
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid(m / BLOCK_SIZE / MAT_SIZE, k / BLOCK_SIZE / MAT_SIZE);
  RUN_kernel(mat_mul_v3_kernel, grid, block, A_d, B_d, m, n, k, C_d);
  CHECK(cudaMemcpy(C_h, C_d, m * k * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(A_d));
  CHECK(cudaFree(B_d));
  CHECK(cudaFree(C_d));
  free(B_T_h);
}

#undef BLOCK_SIZE
#undef MAT_SIZE
//v4 : add prefresh

#define BLOCK_SIZE 16
#define MAT_SIZE 8

__global__ void mat_mul_v4_kernel (const float4 *A, const float4 *B, size_t m, size_t n, size_t k, float *C) {
  int subx = threadIdx.x * MAT_SIZE;
  int suby = threadIdx.y * MAT_SIZE;
  int blockx = blockIdx.x * BLOCK_SIZE * MAT_SIZE;
  int blocky = blockIdx.y * BLOCK_SIZE * MAT_SIZE;
  int x = blockx + subx;
  int y = blocky + suby;
  int tid = threadIdx.x * BLOCK_SIZE + threadIdx.y;
  
  float sum[MAT_SIZE][MAT_SIZE];
  for (int i = 0; i < MAT_SIZE; i++) {
    for (int j = 0; j < MAT_SIZE; j++) {
      sum[i][j] = 0;
    }
  }
  __shared__ float4 As[2][BLOCK_SIZE * MAT_SIZE][MAT_SIZE / 4];
  __shared__ float4 Bs[2][BLOCK_SIZE * MAT_SIZE][MAT_SIZE / 4];
  int write_stage_idx = 0;
  if(tid < BLOCK_SIZE * MAT_SIZE * MAT_SIZE / 4) {
    int tx = tid / (MAT_SIZE / 4), ty = tid % (MAT_SIZE / 4);
    As[0][tx][ty] = __ldg(A + (((blockx + tx) * n) >> 2) + ty);
    Bs[0][tx][ty] = __ldg(B + (((blocky + tx) * n) >> 2) + ty);
  }
  
  __syncthreads();
  int i = 0;
  int up = n / MAT_SIZE;
  do {
    int load_stage_idx = write_stage_idx;
    write_stage_idx ^= 1;
    if (i + 1 < up) {
      if(tid < BLOCK_SIZE * MAT_SIZE * MAT_SIZE / 4) {
        int tx = tid / (MAT_SIZE / 4), ty = tid % (MAT_SIZE / 4);
        As[write_stage_idx][tx][ty] = __ldg(A + (((blockx + tx) * n + (i + 1) * MAT_SIZE) >> 2) + ty);
        Bs[write_stage_idx][tx][ty] = __ldg(B + (((blocky + tx) * n + (i + 1) * MAT_SIZE) >> 2) + ty);
      }
    }
    float4 Ass[MAT_SIZE / 4], Bss[MAT_SIZE / 4];
    #pragma unroll
    for (int c = 0; c < MAT_SIZE / 4; c++) {
      #pragma unroll
      for (int a = 0; a < MAT_SIZE; a++) {
        Ass[a] = As[load_stage_idx][subx + a][c];
        Bss[a] = Bs[load_stage_idx][suby + a][c];
      }
      #pragma unroll
      for (int a = 0; a < MAT_SIZE; a++) {
        float4 Asss = Ass[a];
        #pragma unroll
        for (int b = 0; b < MAT_SIZE; b++) {
          float4 Bsss = Bss[b];
          sum[a][b] += Asss.x * Bsss.x + Asss.y * Bsss.y + Asss.z * Bsss.z + Asss.w * Bsss.w;
        }
      }
    }
    __syncthreads();
    i++;
  } while(i < up);
  for (int a = 0; a < MAT_SIZE; a++) {
    for (int b = 0; b < MAT_SIZE; b++) {
      C[(x + a) * k + y + b] = sum[a][b];
    }
  }
}

void mat_mul_v4(const float *A_h, const float *B_h, size_t m, size_t n, size_t k, float *C_h) {
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
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid(m / BLOCK_SIZE / MAT_SIZE, k / BLOCK_SIZE / MAT_SIZE);
  RUN_kernel(mat_mul_v4_kernel, grid, block, A_d, B_d, m, n, k, C_d);
  CHECK(cudaMemcpy(C_h, C_d, m * k * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(A_d));
  CHECK(cudaFree(B_d));
  CHECK(cudaFree(C_d));
  free(B_T_h);
}

#undef BLOCK_SIZE
#undef MAT_SIZE

//v5 : trans A

#define MAT_SIZE 128
#define BLOCK_SIZE 16
__global__ void mat_mul_v5_kernel (const float4 *A, const float4 *B, size_t m, size_t n, size_t k, float4 *C) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int blockx = blockIdx.x * MAT_SIZE;
  int blocky = blockIdx.y * MAT_SIZE;
  float4 sum[8][2] = {{make_float4(0,0,0,0)}};
  __shared__ float4 As[BLOCK_SIZE][MAT_SIZE / 4],Bs[BLOCK_SIZE][MAT_SIZE / 4];
  for (int i = 0; i < n; i += BLOCK_SIZE) {
    As[tx][ty * 2] = A[((i + tx) * m + blockx) / 4 + ty * 2];
    As[tx][ty * 2 + 1] = A[((i + tx) * m + blockx) / 4 + ty * 2 + 1];
    Bs[tx][ty * 2] = B[((i + tx) * k + blocky) / 4 + ty * 2];
    Bs[tx][ty * 2 + 1] = B[((i + tx) * k + blocky) / 4 + ty * 2 + 1];
    __syncthreads();
    for (int j = 0; j < BLOCK_SIZE; j++) {
      float4 Ass[2],Bss[2];
      Ass[0] = As[j][tx], Ass[1] = As[j][tx + MAT_SIZE / 8];
      Bss[0] = Bs[j][ty], Bss[1] = Bs[j][ty + MAT_SIZE / 8];
      #pragma unroll
      for (int a = 0; a < 8; a++) {
        #pragma unroll
        for (int b = 0; b < 8; b++) {
          ((float*)sum[a])[b] += ((float*)Ass)[a] * ((float*)Bss)[b];
        }
      }
    }
    __syncthreads();
  }
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 2; j++) {
      C[((blockx + tx * 4 + (i & 3) + (i >> 2 & 1) * 64) * k + blocky + ty * 4 + j * 64) / 4] = sum[i][j];
    }
  }
}

void mat_mul_v5(const float *A_h, const float *B_h, size_t m, size_t n, size_t k, float *C_h) {
  float *A_T_h = (float*)malloc(m * n * sizeof(float));
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      A_T_h[j * m + i] = A_h[i * n + j];
    }
  }
  float4 *A_d, *B_d, *C_d;
  CHECK(cudaMalloc(&A_d, m * n * sizeof(float)));
  CHECK(cudaMalloc(&B_d, n * k * sizeof(float)));
  CHECK(cudaMalloc(&C_d, m * k * sizeof(float)));
  CHECK(cudaMemcpy(B_d, B_h, n * k * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(A_d, A_T_h, n * m * sizeof(float), cudaMemcpyHostToDevice));
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid(m / MAT_SIZE, k / MAT_SIZE);
  RUN_kernel(mat_mul_v5_kernel, grid, block, A_d, B_d, m, n, k, C_d);
  CHECK(cudaMemcpy(C_h, C_d, m * k * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(A_d));
  CHECK(cudaFree(B_d));
  CHECK(cudaFree(C_d));
  free(A_T_h);
}
#undef MAT_SIZE
#undef BLOCK_SIZE

//v6 : add prefresh

#define MAT_SIZE 128
#define BLOCK_SIZE 16
__global__ void mat_mul_v6_kernel (const float4 *A, const float4 *B, size_t m, size_t n, size_t k, float4 *C) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int blockx = blockIdx.x * MAT_SIZE;
  int blocky = blockIdx.y * MAT_SIZE;
  float4 sum[8][2] = {{make_float4(0,0,0,0)}};
  __shared__ float4 As[2][BLOCK_SIZE * MAT_SIZE / 4],Bs[2][BLOCK_SIZE * MAT_SIZE / 4];
  int write_stage_idx = 0;
  int write_pos_idx = tx * (MAT_SIZE / 4) + ty * 2;
  const float4 *pA = A + ty * 2 + (tx * m + blockx) / 4;
  const float4 *pB = B + ty * 2 + (tx * k + blocky) / 4;
  As[write_stage_idx][write_pos_idx] = __ldg(pA);
  As[write_stage_idx][write_pos_idx + 1] = __ldg(pA + 1);
  Bs[write_stage_idx][write_pos_idx] = __ldg(pB);
  Bs[write_stage_idx][write_pos_idx + 1] = __ldg(pB + 1);
  __syncthreads();
  float4 Ass[2],Bss[2];
  for (int i = 0; i < n; ) {
    int load_stage_idx = write_stage_idx;
    write_stage_idx ^= 1;
    i += BLOCK_SIZE;
    pA += BLOCK_SIZE * m / 4;
    pB += BLOCK_SIZE * k / 4;
    if (i < n) {
      As[write_stage_idx][write_pos_idx] = __ldg(pA);
      As[write_stage_idx][write_pos_idx + 1] = __ldg(pA + 1);
      Bs[write_stage_idx][write_pos_idx] = __ldg(pB);
      Bs[write_stage_idx][write_pos_idx + 1] = __ldg(pB + 1);
    }
    for (int j = 0; j < BLOCK_SIZE; j++) {
      Ass[0] = As[load_stage_idx][j * (MAT_SIZE / 4) + tx], Ass[1] = As[load_stage_idx][j * (MAT_SIZE / 4) + tx + MAT_SIZE / 8];
      Bss[0] = Bs[load_stage_idx][j * (MAT_SIZE / 4) + ty], Bss[1] = Bs[load_stage_idx][j * (MAT_SIZE / 4) + ty + MAT_SIZE / 8];
      #pragma unroll
      for (int a = 0; a < 8; a++) {
        #pragma unroll
        for (int b = 0; b < 8; b++) {
          ((float*)sum[a])[b] += ((float*)Ass)[a] * ((float*)Bss)[b];
        }
      }
    }
    __syncthreads();
  }
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 2; j++) {
      C[((blockx + tx * 4 + (i & 3) + (i >> 2 & 1) * 64) * k + blocky + ty * 4 + j * 64) / 4] = sum[i][j];
    }
  }
}

void mat_mul_v6(const float *A_h, const float *B_h, size_t m, size_t n, size_t k, float *C_h) {
  float *A_T_h = (float*)malloc(m * n * sizeof(float));
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      A_T_h[j * m + i] = A_h[i * n + j];
    }
  }
  float4 *A_d, *B_d, *C_d;
  CHECK(cudaMalloc(&A_d, m * n * sizeof(float)));
  CHECK(cudaMalloc(&B_d, n * k * sizeof(float)));
  CHECK(cudaMalloc(&C_d, m * k * sizeof(float)));
  CHECK(cudaMemcpy(B_d, B_h, n * k * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(A_d, A_T_h, n * m * sizeof(float), cudaMemcpyHostToDevice));
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid(m / MAT_SIZE, k / MAT_SIZE);
  RUN_kernel(mat_mul_v6_kernel, grid, block, A_d, B_d, m, n, k, C_d);
  CHECK(cudaMemcpy(C_h, C_d, m * k * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(A_d));
  CHECK(cudaFree(B_d));
  CHECK(cudaFree(C_d));
  free(A_T_h);
}
#undef MAT_SIZE
#undef BLOCK_SIZE

//v7 : code = copy, de = py

#define MAT_SIZE 128
#define BLOCK_SIZE 16
__global__ void mat_mul_v7_kernel (const float4 *A, const float4 *B, size_t m, size_t n, size_t k, float4 *C) {
  int tx = threadIdx.y;
  int ty = threadIdx.x;
  int blockx = blockIdx.x * MAT_SIZE;
  int blocky = blockIdx.y * MAT_SIZE;
  float4 c[8][2] = {{make_float4(0,0,0,0)}};
  __shared__ float4 smemA[2][BLOCK_SIZE * MAT_SIZE / 4], smemB[2][BLOCK_SIZE * MAT_SIZE / 4];
  float4 ldg_a_reg[2], ldg_b_reg[2];
  float4 a_reg[2][2], b_reg[2][2];
  int write_pos_idx = tx * (MAT_SIZE / 4) + ty * 2;
  const float4 *pA = A + ty * 2 + (tx * m + blockx) / 4;
  const float4 *pB = B + ty * 2 + (tx * k + blocky) / 4;
  ldg_a_reg[0] = __ldg(pA), ldg_a_reg[1] = __ldg(pA + 1);
  ldg_b_reg[0] = __ldg(pB), ldg_b_reg[1] = __ldg(pB + 1);
  smemA[0][write_pos_idx] = ldg_a_reg[0], smemA[0][write_pos_idx + 1] = ldg_a_reg[1];
  smemB[0][write_pos_idx] = ldg_b_reg[0], smemB[0][write_pos_idx + 1] = ldg_b_reg[1];
  int write_stage_idx = 1;
  __syncthreads();
  a_reg[0][0] = smemA[0][tx], a_reg[0][1] = smemA[0][tx + (MAT_SIZE / 8)];
  b_reg[0][0] = smemB[0][ty], b_reg[0][1] = smemB[0][ty + (MAT_SIZE / 8)];
  int i = 0;
  do {
    i += BLOCK_SIZE;
    pA += BLOCK_SIZE * m / 4;
    pB += BLOCK_SIZE * k / 4;
    if (i < n) {
      ldg_a_reg[0] = __ldg(pA), ldg_a_reg[1] = __ldg(pA + 1);
      ldg_b_reg[0] = __ldg(pB), ldg_b_reg[1] = __ldg(pB + 1);
    }
    int load_stage_idx = write_stage_idx ^ 1;
    #pragma unroll
    for (int j = 0; j < BLOCK_SIZE - 1; j++) {
      a_reg[(j + 1) & 1][0] = smemA[load_stage_idx][(j + 1) * (MAT_SIZE / 4) + tx];
      a_reg[(j + 1) & 1][1] = smemA[load_stage_idx][(j + 1) * (MAT_SIZE / 4) + tx + (MAT_SIZE / 8)];
      b_reg[(j + 1) & 1][0] = smemB[load_stage_idx][(j + 1) * (MAT_SIZE / 4) + ty];
      b_reg[(j + 1) & 1][1] = smemB[load_stage_idx][(j + 1) * (MAT_SIZE / 4) + ty + (MAT_SIZE / 8)];
      #pragma unroll
      for (int a = 0; a < 8; a++) {
        #pragma unroll
        for (int b = 0; b < 8; b++) {
          ((float*)c[a])[b] += ((float*)a_reg[j & 1])[a] * ((float*)b_reg[j & 1])[b];
        }
      }
    }
    smemA[write_stage_idx][write_pos_idx] = ldg_a_reg[0], smemA[write_stage_idx][write_pos_idx + 1] = ldg_a_reg[1];
    smemB[write_stage_idx][write_pos_idx] = ldg_b_reg[0], smemB[write_stage_idx][write_pos_idx + 1] = ldg_b_reg[1];
    __syncthreads();
      write_stage_idx ^= 1;
    a_reg[0][0] = smemA[load_stage_idx ^ 1][tx], a_reg[0][1] = smemA[load_stage_idx ^ 1][tx + (MAT_SIZE / 8)];
    b_reg[0][0] = smemB[load_stage_idx ^ 1][ty], b_reg[0][1] = smemB[load_stage_idx ^ 1][ty + (MAT_SIZE / 8)];
    #pragma unroll
    for (int a = 0; a < 8; a++) {
      #pragma unroll
      for (int b = 0; b < 8; b++) {
        ((float*)c[a])[b] += ((float*)a_reg[1])[a] * ((float*)b_reg[1])[b];
      }
    }
  } while(i < n);
  float4 *pC = C + blockx / 4 * k + tx * k + blocky / 4 + ty;
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    pC[i * k / 4] = c[i][0];
    pC[i * k / 4 + 16] = c[i][1];
  }
  pC += k * 16;
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    pC[i * k / 4] = c[i | 4][0];
    pC[i * k / 4 + 16] = c[i | 4][1];
  }
}

void mat_mul_v7(const float *A_h, const float *B_h, size_t m, size_t n, size_t k, float *C_h) {
  float *A_T_h = (float*)malloc(m * n * sizeof(float));
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      A_T_h[j * m + i] = A_h[i * n + j];
    }
  }
  float4 *A_d, *B_d, *C_d;
  CHECK(cudaMalloc(&A_d, m * n * sizeof(float)));
  CHECK(cudaMalloc(&B_d, n * k * sizeof(float)));
  CHECK(cudaMalloc(&C_d, m * k * sizeof(float)));
  CHECK(cudaMemcpy(B_d, B_h, n * k * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(A_d, A_T_h, n * m * sizeof(float), cudaMemcpyHostToDevice));
  Timer t;
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid(m / MAT_SIZE, k / MAT_SIZE);
  RUN_kernel(mat_mul_v7_kernel, grid, block, A_d, B_d, m, n, k, C_d);
  CHECK(cudaMemcpy(C_h, C_d, m * k * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(A_d));
  CHECK(cudaFree(B_d));
  CHECK(cudaFree(C_d));
  free(A_T_h);
}
#undef MAT_SIZE
#undef BLOCK_SIZE

#define MAT_SIZE 128
#define BLOCK_SIZE 16
#define TILE_K 8
__global__ void mat_mul_v8_kernel (const float4 *A, const float4 *B, size_t m, size_t n, size_t k, float4 *C) {
  int tx = threadIdx.y;
  int ty = threadIdx.x;
  int blockx = blockIdx.x * MAT_SIZE;
  int blocky = blockIdx.y * MAT_SIZE;
  __shared__ float4 smemA[2][TILE_K * MAT_SIZE / 4];
  __shared__ float4 smemB[2][TILE_K * MAT_SIZE / 4];
  float4 c[8][2] = {{make_float4(0,0,0,0)}};
  float4 a_reg[2][2];
  float4 b_reg[2][2];
  float4 ldg_a_reg;
  float4 ldg_b_reg;
  int write_pos_idx = tx * 16 + ty;
  int tx32 = write_pos_idx / 32;
  int ty32 = write_pos_idx % 32;
  A += ty32 + (tx32 * m + blockx) / 4;
  B += ty32 + (tx32 * k + blocky) / 4;
  smemA[0][write_pos_idx] = __ldg(A);
  smemB[0][write_pos_idx] = __ldg(B);
  __syncthreads();
  a_reg[0][0] = smemA[0][tx];
  a_reg[0][1] = smemA[0][tx + (MAT_SIZE / 8)];
  b_reg[0][0] = smemB[0][ty];
  b_reg[0][1] = smemB[0][ty + (MAT_SIZE / 8)];
  int write_stage_idx = 1;
  int i = 0;
  do {
    i += TILE_K;
    A += (TILE_K / 4) * m;
    B += (TILE_K / 4) * k;
    if (i < n) {
      ldg_a_reg = __ldg(A), ldg_b_reg = __ldg(B);
    }
    int load_stage_idx = write_stage_idx ^ 1;
    #pragma unroll
    for (int j = 0; j < TILE_K - 1; j++) {
      a_reg[(j + 1) & 1][0] = smemA[load_stage_idx][(j + 1) * (MAT_SIZE / 4) + tx];
      a_reg[(j + 1) & 1][1] = smemA[load_stage_idx][(j + 1) * (MAT_SIZE / 4) + tx + (MAT_SIZE / 8)];
      b_reg[(j + 1) & 1][0] = smemB[load_stage_idx][(j + 1) * (MAT_SIZE / 4) + ty];
      b_reg[(j + 1) & 1][1] = smemB[load_stage_idx][(j + 1) * (MAT_SIZE / 4) + ty + (MAT_SIZE / 8)];
      #pragma unroll
      for (int a = 0; a < 8; a++) {
        #pragma unroll
        for (int b = 0; b < 8; b++) {
          ((float*)c[a])[b] += ((float*)a_reg[j & 1])[a] * ((float*)b_reg[j & 1])[b];
        }
      }
    }
    smemA[write_stage_idx][write_pos_idx] = ldg_a_reg, smemB[write_stage_idx][write_pos_idx] = ldg_b_reg;
    __syncthreads();
    write_stage_idx ^= 1;
    a_reg[0][0] = smemA[load_stage_idx ^ 1][tx], a_reg[0][1] = smemA[load_stage_idx ^ 1][tx + (MAT_SIZE / 8)];
    b_reg[0][0] = smemB[load_stage_idx ^ 1][ty], b_reg[0][1] = smemB[load_stage_idx ^ 1][ty + (MAT_SIZE / 8)];
    #pragma unroll
    for (int a = 0; a < 8; a++) {
      #pragma unroll
      for (int b = 0; b < 8; b++) {
        ((float*)c[a])[b] += ((float*)a_reg[1])[a] * ((float*)b_reg[1])[b];
      }
    }
  } while(i < n);
  C += blockx / 4 * k + tx * k + blocky / 4 + ty;
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    C[i * k / 4] = c[i][0];
  }
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    C[i * k / 4 + 16] = c[i][1];
  }
  C += k * 16;
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    C[i * k / 4] = c[i | 4][0];
  }
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    C[i * k / 4 + 16] = c[i | 4][1];
  }
}

void mat_mul_v8(const float *A_h, const float *B_h, size_t m, size_t n, size_t k, float *C_h) {
  float *A_T_h = (float*)malloc(m * n * sizeof(float));
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      A_T_h[j * m + i] = A_h[i * n + j];
    }
  }
  float4 *A_d, *B_d, *C_d;
  CHECK(cudaMalloc(&A_d, m * n * sizeof(float)));
  CHECK(cudaMalloc(&B_d, n * k * sizeof(float)));
  CHECK(cudaMalloc(&C_d, m * k * sizeof(float)));
  CHECK(cudaMemcpy(B_d, B_h, n * k * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(A_d, A_T_h, n * m * sizeof(float), cudaMemcpyHostToDevice));
  Timer t;
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid(m / MAT_SIZE, k / MAT_SIZE);
  RUN_kernel(mat_mul_v8_kernel, grid, block, A_d, B_d, m, n, k, C_d);
  CHECK(cudaMemcpy(C_h, C_d, m * k * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(A_d));
  CHECK(cudaFree(B_d));
  CHECK(cudaFree(C_d));
  free(A_T_h);
}
#undef MAT_SIZE
#undef BLOCK_SIZE

signed main(){
  int m = 1 << 12, n = 1 << 12, k = 1 << 12;
  float *A, *B, *STD, *OUT;
  A = (float*)malloc(m * n * sizeof(float));
  B = (float*)malloc(n * k * sizeof(float));
  STD = (float*)malloc(m * k * sizeof(float));
  OUT = (float*)malloc(m * k * sizeof(float));
  
  initialData(A, m * n);
  initialData(B, n * k);
  // mat_mul_cpu(A, B, m, n, k, STD);
  mat_mul_cub(A, B, m, n, k, OUT);
  // checkResult(STD, OUT ,m * k);
  mat_mul_v1(A, B, m, n, k, OUT);
  // checkResult(STD, OUT ,m * k);
  mat_mul_v2(A, B, m, n, k, OUT);
  // checkResult(STD, OUT ,m * k);
  mat_mul_v3(A, B, m, n, k, OUT);
  // checkResult(STD, OUT ,m * k);
  mat_mul_v4(A, B, m, n, k, OUT);
  // checkResult(STD, OUT ,m * k);
  mat_mul_v5(A, B, m, n, k, OUT);
  // checkResult(STD, OUT ,m * k);
  mat_mul_v6(A, B, m, n, k, OUT);
  // checkResult(STD, OUT ,m * k);
  mat_mul_v7(A, B, m, n, k, OUT);
  // checkResult(STD, OUT ,m * k);
  mat_mul_v8(A, B, m, n, k, OUT);
  // checkResult(STD, OUT ,m * k);
  
  free(A);
  free(B);
  free(STD);
  free(OUT);
}