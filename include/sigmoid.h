#ifndef SIGMOID_H
#define SIGMOID_H
#include <device_matrix.h>
#include <vector>
#include <fstream>
using namespace std;

typedef device_matrix<float> W;

class sigmoid{
public:
	sigmoid();
	sigmoid(W);
	sigmoid(size_t,size_t);
	~sigmoid();
	
	void forward(vector<float>&);
	void backPropagate(vector<float>&);	
		
	void print(ofstream*);
private:
	void rand_init();
	W* _weight;
	vector<float>* _sigout;  //output
	vector<float>* _sigoutdiff; //differentiation of output
};

#endif
