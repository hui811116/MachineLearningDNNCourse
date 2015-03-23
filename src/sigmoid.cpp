#include "sigmoid.h"
#include "util.h"
#include <device_matrix.h>
#include <vector>
#include <fstream>
// nvcc compiler
#include <device_arithmetic.h>
#include <device_math.h>

using namespace std;
//using namespace ext;
using namespace func;

typedef device_matrix<float> mat;
typedef thrust::device_vector<float> vec;

Sigmoid::Sigmoid(){
	_weight.resize(1,2);
	_sigout.resize(2,1);
	_input.resize(1,1);
	_weight->fillwith(0);
}
Sigmoid::Sigmoid(const mat& m){
	_weight=m;
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
	// 
	if(train){
		_input = in;
		_sigout = _weight * (*_inp);	
	}
	delete _inp;
}

// assume error pass through var "delta"
Sigmoid::backPropagate(mat& out, const mat& delta, float rate){
	mat _tmp( (~_weight) * delta);
	out= _tmp & sigmoid(_sigout) & (1-sigmoid(_sigout));
	// update weight
	/*
	float* h_data = new float(_weight->size());
	float* d_data = out.getData();
	
	for(int i=0 ;i<_weight->size();++i)
		h_data[i]=d_data[i % out.getRows()];
	*/
	mat _inp(_input.getRows()+1,1,1);
	float* h_data = _input.getData();
	float* d_data = _inp.getData();
	for(int i=0;i<_input.size();++i)
		d_data[i]=h_data[i];
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
	T* h_data = new T [_s];
	for (size_t i=0; i<_s; ++i)
		h_data[i]=rand() / (float) RAND_MAX;
	cudaMemcpy(_weight.getData(), h_data, _weight.size() * sizeof(float), cudaMemcpyHostToDevice);
	delete [] h_data;
}


