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
	Sigmoid(const mat&);
	Sigmoid(size_t, size_t);
	~Sigmoid();
	
	void forward(mat&, const mat&);
	void backPropagate(mat&, const mat&);	
		
	void print(ofstream*);
private:
	void rand_init();
	mat* _weight;
	mat* _input; //for backpropagation
	mat* _sigout;  //output
};

#endif
