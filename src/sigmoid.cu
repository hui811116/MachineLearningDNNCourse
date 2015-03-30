#include "sigmoid.h"
#include <device_matrix.h>
#include <vector>
#include <fstream>
#include <cassert>
#include <cmath>
#include <ctime>
#include <cstdlib>
// nvcc compiler
#include <device_arithmetic.h>
#include <device_math.h>

using namespace std;

typedef device_matrix<float> mat;

Sigmoid::Sigmoid(){
	_weight.resize(1,2);
	_input.resize(1,1);
	_weight.fillwith(0);
}
Sigmoid::Sigmoid(const mat& w){
	_weight=w;
	_input.resize(_weight.getCols()-1,1);
}
Sigmoid::Sigmoid(size_t out_dim, size_t inp_dim){
	_weight.resize(out_dim,inp_dim+1);  // +1 for bias
	rand_init();
	//_weight/=sqrt(inp_dim);
}

Sigmoid::~Sigmoid(){
}

void Sigmoid::forward(mat& out, const mat& in, bool train){
	mat _inp = mat(in);
	pushOne(_inp);
	//fill with 1 for computation simplicity
	out = ext::sigmoid( (_weight * _inp));
	if(train){
		_input = in;
	}
}

// assume error pass through var "delta"
void Sigmoid::backPropagate(mat& out, const mat& delta, float rate){
	assert( (delta.getRows()==_weight.getRows()) && (delta.getCols()==_input.getCols()) );
	mat withoutBias(_weight.getRows(),_weight.getCols()-1);
	CCE(cudaMemcpy(withoutBias.getData(),_weight.getData(),withoutBias.size() * sizeof(float),cudaMemcpyDeviceToDevice));
	mat _tmp( (~withoutBias) * delta);
	mat one(_input.getRows(),_input.getCols(),1);
	mat diff= (_input) & (one-_input);
	out = diff & _tmp;   // this part need tesing
	// update weight
	mat _inp(_input);
	pushOne(_inp);
	gemm(delta,_inp,_weight,(float)-1.0*rate,(float)1.0,false,true);
	//gemm(delta,_inp,_weight,(float)-1.0*rate/(float)_input.getCols(),(float)1.0,false,true);
}

void Sigmoid::getSigDiff(mat& delta,const mat& error){
	assert( (error.getRows()==_weight.getRows()) && (error.getCols()==_input.getCols()) );
	mat one(_weight.getRows(),_input.getCols(),1);
	mat _inp(_input);
	pushOne(_inp);
	delta = (_weight * _inp);
	mat sig=ext::sigmoid(delta);
	delta = (sig) & (one-sig) & error;
}

void Sigmoid::write(ofstream& out){
	float* h_data = new float[_weight.size()];
	CCE(cudaMemcpy( h_data, _weight.getData(), _weight.size() * sizeof(float), cudaMemcpyDeviceToHost));
    out<<"<sigmoid> "<<_weight.getRows()<<" "<<_weight.getCols()<<endl;
    for(size_t i=0;i<_weight.getRows();++i){
    for(size_t j=0;j<_weight.getCols()-1;++j){
                out<<" "<<h_data[_weight.getRows()*j+i]; 
            }
            out<<endl;
    }
    out<<"<bias> "<<_weight.getRows()<<endl;
    for(size_t t=0;t<_weight.getRows();++t)
                out<<" "<<h_data[_weight.getRows()*(_weight.getCols()-1)+t];
	delete [] h_data;
}

void Sigmoid::print(FILE* fid, int precision, char delimiter){
	float* h_data = new float[_weight.size()];
	CCE(cudaMemcpy( h_data, _weight.getData(), _weight.size() * sizeof(float), cudaMemcpyDeviceToHost));

	char format[16];
	sprintf(format,"%c%%.%de",delimiter,(precision>0)? precision :0);
	
	fprintf(fid,"<sigmoid> %d %d \n",_weight.getRows() ,_weight.getCols()); // <sigmoid> outputDimension inputDimension
	for(size_t i=0;i<_weight.getRows();++i){
		for(size_t j=0;j<_weight.getCols()-1;++j)
			fprintf(fid,format,h_data[j*_weight.getRows()+i]);
		fprintf(fid,"\n");
	}
	
	fprintf(fid,"<bias> %d \n",_weight.getRows()); // <bias> output dimensions
	for(size_t t=0;t<_weight.getRows();++t)
		fprintf(fid,format,h_data[_weight.getRows() * (_weight.getCols()-1) + t]);
	fprintf(fid,"\n");
	
	delete [] h_data;
}
size_t Sigmoid::getInputDim(){
	return _weight.getCols()-1;
}
size_t Sigmoid::getOutputDim(){
	return _weight.getRows();
}
void Sigmoid::rand_init(){
    srand(time(0));
	size_t _s=_weight.size();
	float* h_data = new float [_s];
	for (size_t i=0; i<_s; ++i)
		h_data[i]=(rand() / (float) RAND_MAX) -0.5;
	CCE(cudaMemcpy(_weight.getData(), h_data, _weight.size() * sizeof(float), cudaMemcpyHostToDevice));
	delete [] h_data;
}

void Sigmoid::pushOne(mat& input){
	device_matrix<float> tmp(~input);
    float* h_data = new float [input.size()+input.getCols()];
	CCE(cudaMemcpy(h_data, tmp.getData(), tmp.size() * sizeof(float), cudaMemcpyDeviceToHost));
    for(size_t t=0;t<tmp.getRows();++t)
	h_data[tmp.size()+t]=1;
	tmp.resize(tmp.getRows(),tmp.getCols()+1);
	CCE(cudaMemcpy(tmp.getData(), h_data, tmp.size() * sizeof(float), cudaMemcpyHostToDevice));
    input=~tmp;
	delete [] h_data;
}

