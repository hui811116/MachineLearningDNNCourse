#ifndef SIGMOID_H
#define SIGMOID_H
#include <device_matrix.h>
#include <vector>
#include <fstream>
using namespace std;

typedef device_matrix<float> mat;

class Sigmoid{
public:
	Sigmoid();
	Sigmoid(const mat& w);
	Sigmoid(size_t r, size_t c);
	~Sigmoid();
	
	void forward(mat& out, const mat& in, bool train);
	void backPropagate(mat& out, const mat& delta, float ln);	
	size_t getInputDim();
	size_t getOutputDim();

	void print(ofstream* os);
private:
	void rand_init();
	mat _weight;
	mat _input; //for backpropagation
	mat _sigout;  //output
};

#endif
