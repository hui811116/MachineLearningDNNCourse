#ifndef TRANSFORMS_H
#define TRANSFORMS_H
#include <device_matrix>
#include <fstream>
using namespace std;

typedef device_matrix<float> mat;

class Transforms{
	public:
		Transforms(const Transforms& t);
		virtual void forward(mat& out,const mat& in) = 0;
		virtual void backPropagate(mat& out,const mat& delta,float rate,float momentum)=0;

		void print();
		void write(ofstream& out);

		size_t getInputDim()const;
		size_t getOutputDim()const;
	protected:
		Transforms(const mat& w);
		Transforms(size_t inputdim, size_t outputdim);
		mat _w;
	private:
	
};


class Sigmoid : public Transforms{
public:
	Sigmoid(const Sigmoid& s);
	Sigmoid(const mat& w);
	Sigmoid(size_t inputdim, size_t outputdim);
	void forward(mat& out,const mat& in);
	void backPropagate(mat& out, const mat& delta, float rate,float momentum);

private:
	void rand_init();
	void rand_norm(float var);
};

class Softmax : public Transforms{
public:
	SoftMax(const Softmax& s);
	Softmax(const mat& w);
	Softmax(size_t inputdim,size_t outputdim);
	void forward(mat& out,const mat& in);
	void backPropagete(mat& out, const mat& delta, float rate,float momentum);

private:
	void rand_init();
	void rand_narm(float var);
};

#endif