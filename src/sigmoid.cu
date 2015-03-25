#include "sigmoid.h"
#include <device_matrix.h>
#include <vector>
#include <fstream>
#include <cassert>
// nvcc compiler
#include <device_arithmetic.h>
#include <device_math.h>

using namespace std;
using namespace ext;

typedef device_matrix<float> mat;

Sigmoid::Sigmoid(){
	_weight.resize(1,2);
	_sigout.resize(2,1);
	_input.resize(1,1);
	_weight->fillwith(0);
}
Sigmoid::Sigmoid(const mat& w){
	_weight=w;
	_sigout.resize(_weight.getRows(),1);
	_input.resize(_weight.getCols()-1,1);
}
Sigmoid::Sigmoid(size_t row, size_t col){
	_weight.resize(row,col+1);  // +1 for bias
	_sigout.resize(row,1);
	_input.resize(col,1);
	rand_init();
}
Sigmoid::~Sigmoid(){
}

void Sigmoid::forward(mat& out, const mat& in, bool train){
	//assume in is a vector
	mat* _inp = new mat(in);
	_inp->resize(in.getRows()+1,in.getCols());
	float* h_data=_inp->getData();
	h_data[in.getRows()]=1;
	//fill with 1 for computation simplicity
	out = sigmoid( _weight * (*_inp));
	//if in training mode 
	if(train){
		_input = in;
		_sigout = _weight * (*_inp);	
	}
	delete _inp;
}

// assume error pass through var "delta"
Sigmoid::backPropagate(mat& out, const mat& delta, float rate){
	mat _tmp( (~_weight) * delta);
	out= _tmp & _sigout & (1-_sigout) ;   // this part need tesing
	
	// update weight
	mat _inp(_input);
	_inp.resize(_input.getRows()+1,1);
	float* h_data = _inp.getData();
	h_data[_input.getRows()]=1;
	gemm(out,_inp,_weight,-rate,1.0,false,true);

}

void Sigmoid::print(ofstream& out){
}

size_t Sigmoid::getInputDim(){
	return _weight.getCols()-1;
}
size_t Sigmoid::getOutputDim(){
	return _weight.getRows();
}
void Sigmoid::rand_init(){
	size_t _s=_weight.size();
	float* h_data = new float [_s];
	for (size_t i=0; i<_s; ++i)
		h_data[i]=rand() / (float) RAND_MAX;
	cudaMemcpy(_weight.getData(), h_data, _weight.size() * sizeof(float), cudaMemcpyHostToDevice);
	delete [] h_data;
}

// element-wise operation

