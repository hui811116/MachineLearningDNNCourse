#include "transforms.h"
#include <iostream>
#include <fstream>
#include <cassert>
#include <cstdlib>

#include <device_matrix.h>
#include <device_arithmetic.h>
#include <device_math.h>

<<<<<<< HEAD
//#include <random>

#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

=======
#include <random>
>>>>>>> FETCH_HEAD

using namespace std;
using namespace ext;

<<<<<<< HEAD
typedef device_matrix<float> mat;
/////////////helper functions//////////////////////
=======
//helper functions
>>>>>>> FETCH_HEAD
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
<<<<<<< HEAD
	CCE(cudaMemcpy(h_data,tmp.getData(),tmp.size()*sizeof(float),cudaMemcpyDeviceToHost));
	for(size_t t=0;t<tmp.getRows();++t)
		h_data[tmp.size()+t]=1;
	tmp.resize(tmp.getRows(),tmp.getCols()+1);
	CCE(cudaMemcpy(tmp.getData(),h_data,tmp.size()*sizeof(float),cudaMemcpyHostToDevice));
	in = ~tmp;
	delete [] h_data;
}

template<typename T>
struct linear_index_to_row_index : public thrust::unary_function<T,T>
{
	T C;

	__host__ __device__
	linear_index_to_row_index(T C) : C(C) {}
	
	__host__ __device__
	T operator()(T i)
	{
			return i/C;
	}
};

void substractMaxPerRow(mat& x);
mat getRowMax(mat& C);
__global__ void substract_max_per_row(float* const A,float* const rmax, unsigned int rows , unsigned int cols);

void substractMaxPerRow(mat& x) {
	mat rmax = getRowMax(x);

	const int N = 32;
	dim3 grid;
	grid.x = (unsigned int) ceil((float) x.getCols() / N );
	grid.y = (unsigned int) ceil((float) x.getRows() / N );
	dim3 threads(N,N);

	substract_max_per_row<<<grid, threads>>>(x.getData(),rmax.getData() , x.getRows(),x.getCols());
	CCE(cudaDeviceSynchronize());
}


__global__ void substract_max_per_row(float* const A, float * const rmax, unsigned int rows,unsigned int cols){
	int x = blockIdx.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= cols|| y>= rows)
			return;
	A[x * rows +y] -= rmax[y];
}

mat getRowMax(mat& C)
{
	mat rmax(C.getRows(),1);
	mat At = ~C;
	thrust::device_vector<float>row_indices(C.getRows());
	thrust::device_vector<float>row_results(C.getRows());
	thrust::reduce_by_key
	(thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(C.getCols())),
	 thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(C.getCols())) +C.size(),thrust::device_ptr<float>(At.getData()),row_indices.begin(),
	 thrust::device_ptr<float>(rmax.getData()),thrust::equal_to<float>(),thrust::maximum<float>());
	
	return rmax;
}
//////////////////////////////////////////////
=======
	cudaMemcpy(h_data,tmp.getData();tmp.size()*sizeof(float),cudaMemcpyDeviceToHost);
	for(size_t t=0;t<tmp.getRows();++t)
		h_data[tmp.size()+t]=1;
	tmp.resize(tmp.getRows(),tmp.getCols()+1);
	cudaMemcpy(tmp.getData(),h_data,tmp.size()*sizeof(float),cudaMemcpyHostToDevice);
	in = ~tmp;
}
>>>>>>> FETCH_HEAD
/*
void rand_norm(float var,mat&){}
*/
///
<<<<<<< HEAD
=======
typedef device_matrix<float> mat;
>>>>>>> FETCH_HEAD


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
<<<<<<< HEAD
	float* h_data = new float[_w.size()];
	CCE(cudaMemcpy( h_data, _w.getData(), _w.size() * sizeof(float), cudaMemcpyDeviceToHost));
    out<<"<sigmoid> "<<_w.getRows()<<" "<<_w.getCols() - 1<<endl;
    for(size_t i=0;i<_w.getRows();++i){
    for(size_t j=0;j<_w.getCols()-1;++j){
                out<<" "<<h_data[_w.getRows()*j+i]; 
            }
            out<<endl;
    }
    out<<"<bias> "<<_w.getRows()<<endl;
    for(size_t t=0;t<_w.getRows();++t)
                out<<" "<<h_data[_w.getRows()*(_w.getCols()-1)+t];
=======
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
>>>>>>> FETCH_HEAD
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

<<<<<<< HEAD
Sigmoid::Sigmoid(const Sigmoid& s): Transforms(s){
}
Sigmoid::Sigmoid(const mat& w, const mat& bias): Transforms(w){
}
Sigmoid::Sigmoid(size_t inputdim,size_t outputdim): Transforms(inputdim,outputdim){
=======
Sigmoid::Sigmoid(const Sigmoid& s):_w(s._w),_i(s._i),_pw(s._pw){}
Sigmoid::Sigmoid(const mat& w):_w(w){
	_pw.resize(w.getRows(),w.getCols(),0);
}
Sigmoid::Sigmoid(size_t inputdim,size_t outputdim){
	_w.resize(outputdim,inputdim);
	rand_init(_w);
	_pw.resize(outputdim,inputdim,0);
>>>>>>> FETCH_HEAD
}
void Sigmoid::forward(mat& out,const mat& in,bool train){
	mat _inp(in);
	pushOne(_inp);
	out=sigmoid(_w*_inp);
	if(train){
		_i=in;
	}
}
<<<<<<< HEAD
void Sigmoid::backPropagate(mat& out,const mat& delta, float rate,float momentum){
	assert( (delta.getRows()==_w.getRows()) && (delta.getCols()==_i.getCols()) );
	mat withoutBias(_w.getRows(),_w.getCols()-1);
	CCE(cudaMemcpy(withoutBias.getData(),_w.getData(),withoutBias.size() * sizeof(float),cudaMemcpyDeviceToDevice));
	mat _tmp( ~withoutBias * delta);
	mat one(_i.getRows(),_i.getCols(),1);
	out = _i & (one-_i) & _tmp;   // this part need tesing
	// update weight
	mat _inp(_i);
	pushOne(_inp);
	_pw= delta * ~_inp + _pw * momentum;
	//_w -= _pw * rate;
	//NOTE: below are the case without momentum
	//rate=rate/(float)_input.getCols();
	gemm(delta,_inp,_w,(float)-1.0*rate,(float)1.0,false,true);
}
/*
=======
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

>>>>>>> FETCH_HEAD
Sigmoid& Sigmoid::operator=(const Sigmoid& s){
	_w=s._w;
	_i=s._i;
	_pw=s._pw;
		return *this
}
<<<<<<< HEAD
*/
///////////////////////////////
///////////SOFTMAX/////////////

Softmax::Softmax(const Softmax& s): Transforms(s){
}
Softmax::Softmax(const mat& w, const mat& bias):Transforms(w){
}
Softmax::Softmax(size_t inputdim,size_t outputdim): Transforms(inputdim,outputdim){
}
void Softmax::forward(mat& out,const mat& in,bool train){
	mat inp=in;
	pushOne(inp);
	mat z=~(_w * inp);
	substractMaxPerRow(z);
	z=~z; // transpose to column vectors
	mat p(z.getRows(), z.getCols());
	
	thrust::device_ptr<float> zPtr(z.getData());
	thrust::device_ptr<float> pPtr(p.getData());
	thrust::transform(zPtr, zPtr + z.size(),zPtr, func::exp<float>());

	mat sumOfProb =  (mat(p.getRows(), p.getRows(),0) += 1) * p;
	out.resize(_w.getRows(),in.getCols());
	thrust::device_ptr<float> outptr(out.getData());
	thrust::device_ptr<float> sPtr(sumOfProb.getData());
	thrust::transform(pPtr,pPtr+p.size(), sPtr,outptr,thrust::divides<float>());
=======
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
>>>>>>> FETCH_HEAD

	if(train){
		_i=in;
	}
}

<<<<<<< HEAD
void Softmax::backPropagate(mat& out,const mat& delta,float rate, float momentum){
	mat inp(_i);
	pushOne(inp);	
	_pw=delta * ~inp + _pw * momentum;
	//_w-= _pw * rate;
	//NOTE: eq. below haven't include momentum yet.
	gemm(delta,inp,_w,(float)-1.0*rate,(float)1.0,false,true);
	
}

=======
>>>>>>> FETCH_HEAD
///////////////////////////////
