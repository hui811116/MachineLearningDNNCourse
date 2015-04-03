#include "transforms.h"
#include <iostream>
#include <fstream>
#include <cassert>
#include <cstdlib>

#include <device_matrix.h>
#include <device_arithmetic.h>
#include <device_math.h>

#include <random>

using namespace std;
using namespace ext;

//helper functions
void rand_init(mat& w){
	float* h_data = new float[w.size()];
	for(size_t t=0;t<w.getRows()*(w.getCols()-1);++t)
		h_data[t]=2*rand()/RAND_MAX-1;
	for(size_t t=0;t<w.getRows();++t)
		h_data[t+w.getRows()*(w.getCols()-1)]=0;
	CCE(cudaMemcpy(w.getData(),h_data,w.size()* sizeof(float) , cudaMemcpyHostToDevice));
	delete [] h_data;
}
void pushOne(mat& in){
	mat tmp(~in);
	float* h_data=new float[in.getRows()*(in.getCols()+1)];
	cudaMemcpy(h_data,tmp.getData();tmp.size()*sizeof(float),cudaMemcpyDeviceToHost);
	for(size_t t=0;t<tmp.getRows();++t)
		h_data[tmp.size()+t]=1;
	tmp.resize(tmp.getRows(),tmp.getCols()+1);
	cudaMemcpy(tmp.getData(),h_data,tmp.size()*sizeof(float),cudaMemcpyHostToDevice);
	in = ~tmp;
}
/*
void rand_norm(float var,mat&){}
*/
///
typedef device_matrix<float> mat;


///////TRANSFORMS/////////////

Transforms::Transforms(const Transforms& t):_w(t._w),_i(t._i),_pw(t._pw){}

Transforms::Transforms(const mat& w):_w(w){
	_pw.resize(w.getRows(),w.getCols(),0);
}

Transforms::Transforms(size_t inputdim,size_t outputdim){
	_w.resize(outputdim,inputdim);
	rand_init(_w);
	_pw.resize(outputdim,inputdim,0);
}

size_t Transforms::getInputDim()const{
	return _w.getCols();
}
size_t Transforms::getOutputDim()const{
	return _w.getRows();
}

void Transforms::write(ofstream& out){
	float* h_data = new float[_weight.size()];
	CCE(cudaMemcpy( h_data, _weight.getData(), _weight.size() * sizeof(float), cudaMemcpyDeviceToHost));
    out<<"<sigmoid> "<<_weight.getRows()<<" "<<_weight.getCols() - 1<<endl;
    for(size_t i=0;i<_weight.getRows();++i){
    for(size_t j=0;j<_weight.getCols()-1;++j){
                out<<" "<<h_data[_weight.getRows()*j+i]; 
            }
            out<<endl;
    }
    out<<"<bias> "<<_weight.getRows()<<endl;
    for(size_t t=0;t<_weight.getRows();++t)
                out<<" "<<h_data[_weight.getRows()*(_weight.getCols()-1)+t];
	out << endl;
	delete [] h_data;
}
void Transforms::print(){
	cout<<"Weight matrix: last column is bias"<<endl;
	_w.print();
	cout<<endl;
}
///////////////////////////////
/////////SIGMOID///////////////

Sigmoid::Sigmoid(const Sigmoid& s):_w(s._w),_i(s._i),_pw(s._pw){}
Sigmoid::Sigmoid(const mat& w):_w(w){
	_pw.resize(w.getRows(),w.getCols(),0);
}
Sigmoid::Sigmoid(size_t inputdim,size_t outputdim){
	_w.resize(outputdim,inputdim);
	rand_init(_w);
	_pw.resize(outputdim,inputdim,0);
}
void Sigmoid::forward(mat& out,const mat& in,bool train){
	mat _inp(in);
	pushOne(_inp);
	out=sigmoid(_w*_inp);
	if(train){
		_i=in;
	}
}
void Sigmoid::backPropagetion(mat& out,const mat& delta, float rate,float momentum){
	assert( (delta.getRows()==_weight.getRows()) && (delta.getCols()==_input.getCols()) );
	mat withoutBias(_weight.getRows(),_weight.getCols()-1);
	CCE(cudaMemcpy(withoutBias.getData(),_weight.getData(),withoutBias.size() * sizeof(float),cudaMemcpyDeviceToDevice));
	mat _tmp( ~withoutBias * delta);
	mat one(_input.getRows(),_input.getCols(),1);
	out = _input & (one-_input) & _tmp;   // this part need tesing
	// update weight
	mat _inp(_input);
	pushOne(_inp);
	_pw= delta * ~_inp;
	//NOTE: below are the case without momentum
	//rate=rate/(float)_input.getCols();
	gemm(delta,_inp,_weight,(float)-1.0*rate,(float)1.0,false,true);
}

Sigmoid& Sigmoid::operator=(const Sigmoid& s){
	_w=s._w;
	_i=s._i;
	_pw=s._pw;
		return *this
}
///////////////////////////////
///////////SOFTMAX/////////////

Softmax::Softmax(const Softmax s):_w(s._w),_i(s._i),_pw(s._pw){
}
Softmax::Softmax(const mat& w):_w(w){
	_pw.resize(w.getRows(),w.getCols(),0);
}
Softmax::Softmax(size_t inputdim,size_t outputdim){
	_w.resize(outputdim,inputdim);
	rand_init(_w);
	_pw.resize(outputdim,inputdim);
}
//TODO
void Softmax::forward(mat& out,const mat& in,bool train){
	mat inp=in;
	pushOne(inp);
	mat z=_w * inp;
	//substractMax(z);

	if(train){
		_i=in;
	}
}

///////////////////////////////
