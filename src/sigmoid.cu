#include "sigmoid.h"
#include <device_matrix.h>
#include <vector>
#include <fstream>
#include <cassert>
// nvcc compiler
#include <device_arithmetic.h>
#include <device_math.h>

using namespace std;

typedef device_matrix<float> mat;

Sigmoid::Sigmoid(){
	_weight.resize(1,2);
	_sigout.resize(2,1);
	_input.resize(1,1);
	_weight.fillwith(0);
}
Sigmoid::Sigmoid(const mat& w){
	_weight=w;
	_sigout.resize(_weight.getRows(),1);
	_input.resize(_weight.getCols()-1,1);
}
Sigmoid::Sigmoid(size_t out_dim, size_t inp_dim){
	_weight.resize(out_dim,inp_dim+1);  // +1 for bias
	_sigout.resize(out_dim,1);
	_input.resize(inp_dim,1);
	rand_init();
}

Sigmoid::~Sigmoid(){
}

void Sigmoid::forward(mat& out, const mat& in, bool train){
	//assume in is a vector
	mat _inp = mat(in);
	_inp.resize(in.getRows()+1,in.getCols());
	float* h_data=_inp.getData();
	h_data[in.getRows()]=1;
	//fill with 1 for computation simplicity
	out = ext::sigmoid( (_weight * _inp));
	//if in training mode 
	if(train){
		_input = in;
		_sigout = _weight * (_inp);	
	}
}

// assume error pass through var "delta"
void Sigmoid::backPropagate(mat& out, const mat& delta, float rate){
	mat _tmp( (~_weight) * delta);
	mat one(_tmp.getRows(),_tmp.getCols(),1);
	out= _tmp & _sigout & (one-_sigout) ;   // this part need tesing
	// update weight
	mat _inp(_input);
	_inp.resize(_input.getRows()+1,1);
	float* h_data = _inp.getData();
	h_data[_input.getRows()]=1;
	gemm(out,_inp,_weight,-rate,(float)1.0,false,true);

}

void Sigmoid::write(ostream& out){

}

void Sigmoid::print(){
	_weight.print();
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

