#ifndef TRANSFORMS_H
#define TRANSFORMS_H
<<<<<<< HEAD
#include <device_matrix.h>
=======
#include <device_matrix>
>>>>>>> FETCH_HEAD
#include <fstream>
using namespace std;

typedef device_matrix<float> mat;

class Transforms{
	public:
		Transforms(const Transforms& t);
<<<<<<< HEAD
		virtual void forward(mat& out,const mat& in,bool train) = 0;
		virtual void backPropagate(mat& out,const mat& delta,float rate,float momentum) = 0;
=======
		virtual void forward(mat& out,const mat& in) = 0;
		virtual void backPropagate(mat& out,const mat& delta,float rate,float momentum)=0;
>>>>>>> FETCH_HEAD

		void print();
		void write(ofstream& out);

		size_t getInputDim()const;
		size_t getOutputDim()const;
	protected:
		Transforms(const mat& w);
		Transforms(size_t inputdim, size_t outputdim);
		mat _w;
<<<<<<< HEAD
		mat _i;
		mat _pw;
	private:
=======
	private:
	
>>>>>>> FETCH_HEAD
};


class Sigmoid : public Transforms{
<<<<<<< HEAD
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
=======
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
>>>>>>> FETCH_HEAD
