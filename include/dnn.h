#ifndef DNN_H
#define DNN_H
#include <vector>
#include <string>
//#include "sigmoid.h"
#include "dataset.h"

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
	DNN(const string& fn);
	~DNN();

//	void train(Dataset&, Method);
//	void predict(Dataset&, vector<float>&);

	void save(const string& fn);

private:
	bool feedForward(vector<float>& input);
	bool backPropagate();

	size_t _inputDimension;
	size_t _outputDimension;
	float _learningRate;
//	vector<Sigmoid>* _layer;
	Dataset& _inputSet;
    Dataset& _validSet;
};


#endif
