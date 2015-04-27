#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <device_matrix.h>

#include <device_arithmetic.h>
#include <device_math.h>
#include "sigmoid.h"

#include <cublas_v2.h>
#include <helper_cuda.h>
#include <cuda_memory_manager.h>
#include "parser.h"


using namespace std;

typedef device_matrix<float> mat;

template <typename T>
void randomInit(device_matrix<T>& m) {
  T* h_data = new T [m.size()];
  for (int i=0; i<m.size(); ++i)
    h_data[i] = rand() / (T) RAND_MAX;
  cudaMemcpy(m.getData(), h_data, m.size() * sizeof(T), cudaMemcpyHostToDevice);
  delete [] h_data;
}

template <typename T>
void pushOne(device_matrix<T>& m) {
  device_matrix<T> tmp(~m);
  T* h_data = new T [m.size()+m.getCols()];
  cudaMemcpy(h_data, tmp.getData(), tmp.size() * sizeof(T), cudaMemcpyDeviceToHost);
  tmp.resize(tmp.getRows(),tmp.getCols()+1);
  for(size_t t=0;t<tmp.getRows();++t)
  h_data[m.size()+t]=1;
  cudaMemcpy(tmp.getData(), h_data, tmp.size() * sizeof(T), cudaMemcpyHostToDevice);
  m=~tmp;
  delete [] h_data;
}

int main(int argc,char** argv){

PARSER p;


srand(time(0));

mat A(5,8),B(8,5);
randomInit(A);
randomInit(B);

//testing element-wise operation

mat C(8,2), D(8,2,2.5);
randomInit(C);
randomInit(D);

//testing sigmoid function

Sigmoid n1(5,5);

C.resize(8,3);
randomInit(C);

printf("testing push one \n");
pushOne(C);
C.print();

printf("testing ext::sigmoid\n");
(ext::sigmoid(C)).print();

n1.print();

printf("minus const num\n");
C.print();
printf("\n");
(C-1).print();

A.resize(5,8);B.resize(5,8);
randomInit(A);randomInit(B);

C.resize(5,5);
randomInit(C);
gemm(A,B,C,(float)-1,(float)1,false,true);
printf("C=\n");
C.print();

return 0;
}
