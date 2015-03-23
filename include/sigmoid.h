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
	
	void forward(vector<float>&);
	void backPropagate(vector<float>&);	
		
	void print(ofstream*);
private:
	void rand_init();
	mat* _weight;
	vector<float>* _sigout;  //output
	vector<float>* _sigoutdiff; //differentiation of output
};

#endif
