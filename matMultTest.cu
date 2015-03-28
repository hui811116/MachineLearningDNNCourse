#include <iostream>
#include <vector>
#include <device_matrix.h>

#include <device_arithmetic.h>
#include <device_math.h>
#include "sigmoid.h"

#include <cublas_v2.h>
#include <helper_cuda.h>
#include <cuda_memory_manager.h>

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

mat A(5,8),B(8,5);
randomInit(A);
randomInit(B);

printf("A=\n");
A.print();
printf("B=\n");
B.print();

printf("A * B= \n"); (A*B).print();

//testing element-wise operation

mat C(8,2), D(8,2,2.5);
randomInit(C);
randomInit(D);

printf("C=\n");
C.print();
printf("C*2=\n");
(C * 2).print();
printf("D=\n");
D.print();

printf("C & D= \n"); (C&D).print();

//testing sigmoid function

Sigmoid n1(5,5);

//float**

float** _fptr=new float*[10];
for(size_t t=0;t<10;++t){
	_fptr[t]=new float[20];
}

for(size_t t=0;t<10;++t){
	for(size_t k=0;k<20;++k)
		_fptr[t][k]=t+100*k;
}

float* test=_fptr[2];

for(size_t t=0;t<20;++t){
	cout<<" "<<test[t];
	if(t+1%5==0)
		cout<<endl;
}
cout<<endl;

for(size_t t=0;t<10;++t)
	delete [] _fptr[t];
delete [] _fptr;

C.resize(8,3);
printf("C=\n");
C.print();

printf("testing push one \n");
pushOne(C);
C.print();

printf("testing ext::sigmoid\n");
(ext::sigmoid(C)).print();

n1.print();
C.print();
(C-1).print();

A.resize(5,8);B.resize(8,5);
randomInit(A);randomInit(B);
C.resize(5,5);
randomInit(C);
printf("C=\n");
C.print();
printf("A*B=\n");
mat tem=A*B;
tem.print();
gemm(A,B,C,(float)1,(float)2,false,false);
printf("C=\n");
C.print();


return 0;
}
