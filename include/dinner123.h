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
  double epsilon=1.0E-8;
  for(int i=0;i<N;i++)
  {
    if(abs(A[i]-B[i])>epsilon)
    {
      printf("Results don\'t match!\n");
      printf("%f(A[%d] )!= %f(B[%d])\n",A[i],i,B[i],i);
      return;
    }
  }
  printf("Check result success!\n");
}
#endif