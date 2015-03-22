#ifndef DNN_H
#define DNN_H
#include <vector>
#include "sigmoid.h"
//#include "dataset.h"
#include <fstream>

using namespace std;
enum Method{
	ALL, 
	BATCH, 
	ONE
};

class DNN{
public:
	DNN();
	DNN(Dataset&);
	Dnn(ifstream*);
	~Dnn();

	void train(Dataset, Method);
	void predict(Dataset, vector<float>&);

	void save(ofstream&);

private:
	bool feedForward(Dataset, vector<float>&);
	bool backPropagate(Dataset);

	vector<Sigmoid>* _layer;
};


#endif
