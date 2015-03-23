#ifndef DNN_H
#define DNN_H
#include <vector>
#include <string>
#include "sigmoid.h"
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
	DNN(Dataset& data, size_t numOfLayer, float learningRate, const vector<size_t>& v, Method method);
//	DNN(const string& fn);
	~DNN();

	void train();
	vector<float>* predict(const vector<float>& inputVec);

	void save(const string& fn);

private:
	bool feedForward(const vector<float>& input);
	bool backPropagate(const vector<float>& error);

	Dataset& _data;
	size_t _inputDimension;
	size_t _numOfLayer;
	size_t _outputDimension;
	float _learningRate;
	Method _method;
	vector<Sigmoid*>* _layers;
	vector<float>* _validateAccuracy;
};


#endif
