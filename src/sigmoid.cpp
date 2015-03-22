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
	_sigout = new vector<float>;
	_sigoutdiff = new vector<float>;
	_weight->fillwith(0);
}
Sigmoid::Sigmoid(const mat& m){
	_weight = new mat(m);
	_sigout = new vector<float>;
	_sigoutdiff = new vector<float>;
}
Sigmoid::Sigmoid(size_t row, size_t col){
	_weight = new mat(row,col+1);  // +1 for bias
	_sigout = new vector<float>;
	_sigoutdiff = new vector<float>;
	rand_init();
}
Sigmoid::~Sigmoid(){
	delete [] _weight;
	_sigout.clear();_sigoutdiff.clear();
	delete _sigout;
	delete _sigoutdiff;
}

void Sigmoid::forward(vector<float>& input){
	input.push_back(1);
	vec* _tmp = new vec(input);
	vec* _o=new vec(sigmoid(toStlVector(*(_weight) * *(_tmp))));
	input.assign(toStlVector(*_o));
	_sigout->assign(input);
	_sigoutdiff->assign(toStlVector(*_o & (1- *_o));
	delete _tmp;
	delete _o;
}
void Sigmoid::backPropagate(vector<float>& grad){

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


