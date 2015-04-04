#ifndef TRANSFORMS_H
#define TRANSFORMS_H
#include <device_matrix.h>
#include <fstream>
using namespace std;

typedef device_matrix<float> mat;

class Transforms{
	public:
		Transforms(const Transforms& t);
		virtual void forward(mat& out,const mat& in,bool train) = 0;
		virtual void backPropagate(mat& out,const mat& delta,float rate,float momentum) = 0;

		void print();
		void write(ofstream& out);

		size_t getInputDim()const;
		size_t getOutputDim()const;
	protected:
		Transforms(const mat& w);
		Transforms(size_t inputdim, size_t outputdim);
		mat _w;
		mat _i;
		mat _pw;
	private:
};


class Sigmoid : public Transforms{
	public:
	Sigmoid(const Sigmoid& s);
	Sigmoid(const mat& w, const mat& bias);
	Sigmoid(size_t inputdim, size_t outputdim);
	virtual void forward(mat& out,const mat& in,bool train);
	virtual void backPropagate(mat& out, const mat& delta, float rate,float momentum);
//	Sigmoid& operator = (const Sigmoid s);

	private:
};

class Softmax : public Transforms{
	public:
	Softmax(const Softmax& s);
	Softmax(const mat& w, const mat& bias);
	Softmax(size_t inputdim,size_t outputdim);
	virtual void forward(mat& out,const mat& in,bool train);
	virtual void backPropagate(mat& out, const mat& delta, float rate,float momentum);

	private:
};

#endif
