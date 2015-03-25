#include <iostream>
#include <vector>
#include <device_matrix.h>

#include <device_arithmetic.h>
#include <device_math.h>
#include "sigmoid.h"

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

int main(){

mat A(5,8),B(8,5);
randomInit(A);
randomInit(B);

printf("A=\n");
A.print();
printf("B=\n");
B.print();

printf("A * B= \n"); (A*B).print();

//testing element-wise operation

mat C(8,1), D(8,1);
randomInit(C);
randomInit(D);

printf("C=\n");
C.print();
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

return 0;
}
