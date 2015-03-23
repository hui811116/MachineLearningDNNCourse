#include "sigmoid.h"
#include "util.h"
#include <device_matrix.h>
#include <vector>
#include <fstream>
// nvcc compiler
#include <device_arithmetic.h>
#include <device_math.h>


using namespace std;
using namespace ext;

typedef device_matrix<float> mat;
typedef thrust::device_vector<float> vec;

Sigmoid::Sigmoid(){
	_weight = new mat(1,2);
	_sigout = new mat(2,1);
	_input = new mat(1,1);
	_weight->fillwith(0);
}
Sigmoid::Sigmoid(const mat& m){
	_weight = new mat(m);
	_sigout = new mat(_weight->getRows(),1);
	_input = new mat(_weight->getCols()-1,1);
}
Sigmoid::Sigmoid(size_t row, size_t col){
	_weight = new mat(row,col+1);  // +1 for bias
	_sigout = new mat(row,1);
	_input = new mat(col,1);
	rand_init();
}
Sigmoid::~Sigmoid(){
	delete _weight;
	delete _sigout;
	delete _input;

}

void Sigmoid::forward(const mat& in, mat& out){
	//assume in is a vector
	out = sigmoid( *_weight * in);
}

// assume error pass through var "delta"
Sigmoid::backPropagate(const mat& err, mat& out){
	
}

void Sigmoid::print(ofstream& out){
}

void Sigmoid::rand_init(){
	size_t _s=_weight->size();
	T* h_data = new T [_s];
	for (size_t i=0; i<_s; ++i)
		h_data[i]=rand() / (float) RAND_MAX;
	cudaMemcpy(_weight->getData(), h_data, _weight->size() * sizeof(float), cudaMemcpyHostToDevice);
	delete [] h_data;
}


