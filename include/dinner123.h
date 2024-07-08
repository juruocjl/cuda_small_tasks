#ifndef DINNER123_H
#define DINNER123_H

#include <cstdio>
#include <cuda_runtime.h>
#define CHECK(call)\
{\
  const cudaError_t error=call;\
  if(error!=cudaSuccess)\
  {\
      printf("ERROR: %s:%d,",__FILE__,__LINE__);\
      printf("code:%d,reason:%s\n",error,cudaGetErrorString(error));\
      exit(1);\
  }\
}
#define CUBLAS_CHECK(call)\
{\
  const cublasStatus_t  error=call;\
  if(error!=CUBLAS_STATUS_SUCCESS)\
  {\
      printf("ERROR: %s:%d,",__FILE__,__LINE__);\
      exit(1);\
  }\
}

namespace dinner123{
  float last_time;
  bool out_time = 1;
}
#define PrintTime() {\
dinner123::last_time = t.elapsed();\
if(dinner123 :: out_time)printf("%s use %lf ms\n", __func__, dinner123::last_time);\
}
#define RUN(X, Times) {\
double sum = 0;\
dinner123 :: out_time = 0;\
for(int i = 0; i < Times; i++) {X; sum += dinner123 :: last_time;}\
dinner123 :: out_time = 1;\
printf(#X " avg time %.5lf ms\n", sum / Times);\
}

#define RUN_kernel(X, Times) {\
double sum = 0;\
cudaTimer t;\
for(int i = 0; i < Times; i++){t.reset(); X; sum += t.elapsed();}\
printf("%s kernel avg time %.5lf ms after %d tests\n", __func__, sum / Times, Times);\
X;\
}

#include<chrono>
using namespace std;
using namespace std::chrono;
class Timer
{
public:
    Timer() : m_begin(high_resolution_clock::now()) {}
    void reset() { m_begin = high_resolution_clock::now(); }
    //默认输出毫秒
    float elapsed() const
    {
        return duration_cast<chrono::milliseconds>(high_resolution_clock::now() - m_begin).count();
    }
private:
    time_point<high_resolution_clock> m_begin;
};

class cudaTimer
{
public:
    cudaTimer() {
        // Allocate CUDA events that we'll use for timing
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
      }
    void reset() {CHECK(cudaEventRecord(start, NULL)); }
    //默认输出毫秒
    float elapsed() const
    {
        CHECK(cudaEventRecord(stop, NULL));
        CHECK(cudaEventSynchronize(stop));
        float milliseconds = 0;
        CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        return milliseconds;
    }
private:
    cudaEvent_t start, stop;
};

void checkResult(float * A,float * B,const int N)
{
  double epsilon=1.0E-5;
  for(int i=0;i<N;i++)
  {
    if(abs(A[i]-B[i]) > epsilon && abs(A[i]-B[i])/max(abs(A[i]), abs(B[i])) > epsilon)
    {
      printf("Results don\'t match!\n");
      printf("%f(A[%d])!= %f(B[%d])\n",A[i],i,B[i],i);
      return;
    }
  }
  printf("Check result success!\n");
}

#include <vector>
void checkResult(std::vector<int>&A,std::vector<int>&B)
{
  size_t N=A.size();
  if (A.size() != B.size()) {
    printf("Results don\'t match!\n");
    printf("A and B is not same size\n");
  }
  for(int i=0;i<N;i++)
  {
    if(A[i] != B[i])
    {
      printf("Results don\'t match!\n");
      printf("%d(A[%d])!= %d(B[%d])\n",A[i],i,B[i],i);
      return;
    }
  }
  printf("Check result success!\n");
}
#include <random>

void initialData(float* ip,int size)
{
  std::mt19937 rnd(chrono::steady_clock::now().time_since_epoch().count());
  for(int i=0;i<size;i++)
  {
    ip[i]=(float)(rnd()&0xffff)/1000.0f;
  }
}
void initialData(int* ip, int size)
{
  std::mt19937 rnd(chrono::steady_clock::now().time_since_epoch().count());
	for (int i = 0; i<size; i++)
	{
		ip[i] = int(rnd()&0xff);
	}
}


#endif