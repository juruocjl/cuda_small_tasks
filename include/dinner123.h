#ifndef DINNER123_H
#define DINNER123_H

#include <cstdio>

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

#include<chrono>
using namespace std;
using namespace std::chrono;
class Timer
{
public:
    Timer() : m_begin(high_resolution_clock::now()) {}
    void reset() { m_begin = high_resolution_clock::now(); }
    //默认输出毫秒
    int64_t elapsed() const
    {
        return duration_cast<chrono::milliseconds>(high_resolution_clock::now() - m_begin).count();
    }
private:
    time_point<high_resolution_clock> m_begin;
};

void checkResult(float * A,float * B,const int N)
{
  double epsilon=1.0E-5;
  for(int i=0;i<N;i++)
  {
    if(abs(A[i]-B[i]) > epsilon && abs(A[i]-B[i])/max(abs(A[i]), abs(B[i])) > epsilon)
    {
      printf("Results don\'t match!\n");
      printf("%f(A[%d] )!= %f(B[%d])\n",A[i],i,B[i],i);
      return;
    }
  }
  printf("Check result success!\n");
}

#include <random>

void initialData(float* ip,int size)
{
  std::mt19937 rnd(std::random_device{}());
  for(int i=0;i<size;i++)
  {
    ip[i]=(float)(rnd()&0xffff)/1000.0f;
  }
}
void initialData(int* ip, int size)
{
  std::mt19937 rnd(std::random_device{}());
	for (int i = 0; i<size; i++)
	{
		ip[i] = int(rnd()&0xff);
	}
}

#endif