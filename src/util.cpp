#include "mynngen.h"
#include <device_matrix.h>

typedef device_matrix<float> mat;

myNnGen gn(0,0.2);

void rand_init(mat& w,float range){
	float* h_data = new float[w.size()];
	for(size_t t=0;t<w.getRows()*(w.getCols()-1);++t)
		h_data[t]=2*range*rand()/(float)RAND_MAX - range;
	for(size_t t=0;t<w.getRows();++t)
		h_data[t+w.getRows()*(w.getCols()-1)]=0;
	CCE(cudaMemcpy(w.getData(),h_data,w.size()* sizeof(float) , cudaMemcpyHostToDevice));
	delete [] h_data;
}
void rand_norm(mat& w,myNnGen& ran){
	float* h_data = new float[w.size()];
	for(size_t t=0;t<w.getRows()*(w.getCols()-1);++t)
		h_data[t]=ran();
	for(size_t t=0;t<w.getRows();++t)
		h_data[t+w.getRows()*(w.getCols()-1)]=0;
	CCE(cudaMemcpy(w.getData(),h_data,w.size()* sizeof(float) , cudaMemcpyHostToDevice));
	delete [] h_data;
}


void pushOne(mat& in){
	mat tmp(~in);
	float* h_data=new float[(in.getRows()+1)*in.getCols()];
	CCE(cudaMemcpy(h_data,tmp.getData(),tmp.size()*sizeof(float),cudaMemcpyDeviceToHost));
	for(size_t t=0;t<tmp.getRows();++t)
		h_data[tmp.size()+t]=1;
	tmp.resize(tmp.getRows(),tmp.getCols()+1);
	CCE(cudaMemcpy(tmp.getData(),h_data,tmp.size()*sizeof(float),cudaMemcpyHostToDevice));
	in = ~tmp;
	delete [] h_data;
}

