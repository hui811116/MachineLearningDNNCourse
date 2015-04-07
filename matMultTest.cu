#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <device_matrix.h>

#include <device_arithmetic.h>
#include <device_math.h>

#include <cublas_v2.h>
#include <helper_cuda.h>
#include <cuda_memory_manager.h>
#include "parser.h"
#include "transforms.h"
#include <random>

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


default_random_engine gen((unsigned)time(NULL));
size_t dim=500;
float n=0.02;
srand(time(0));

mat A(5,8),B(8,5);
randomInit(A);
randomInit(B);

//testing element-wise operation

mat C(8,2), D(8,2,2.5);
randomInit(C);
randomInit(D);

//testing sigmoid function

printf("divide by a const num\n");
C.print();
printf("\n");
(C*n/(float)dim).print();

mat in(10,3);
randomInit(in);
Softmax s1(10,10);
mat out;
s1.forward(out,in,true);
cout<<"out"<<endl;
out.print();
mat bk;
s1.backPropagate(bk,out,0.02,0);
cout<<"bk="<<endl;
bk.print();

Sigmoid s2(10,10);
cout<<"testing sigmoid"<<endl;
s2.forward(out,in,true);
cout<<"out"<<endl;
out.print();
s2.backPropagate(bk,out,0.02,0);
cout<<"bk="<<endl;
bk.print();

cout<<"testing normal distribution"<<endl;
Sigmoid sss(5,6);
cout<<endl;
Sigmoid ss2(5,6);
cout<<endl;
Softmax s12(5,6);
cout<<endl;
return 0;
}
