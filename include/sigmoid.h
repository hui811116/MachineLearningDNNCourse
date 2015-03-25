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
	Sigmoid(size_t row, size_t col);
	~Sigmoid();
	
	void forward(mat& out, const mat& in, bool train);
	void backPropagate(mat& out, const mat& delta, float rate);	
	size_t getInputDim();
	size_t getOutputDim();

	void print(ofstream* out);
private:
	void rand_init();
	mat operator &(const mat& lv, const mat& rv);

	mat _weight;
	mat _input; //for backpropagation
	mat _sigout;  //output
};

#endif
