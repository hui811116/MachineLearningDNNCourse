#ifndef DNN_H
#define DNN_H
#include <vector>
#include <string>
#include <device_matrix.h>
#include "sigmoid.h"
#include "dataset.h"
using namespace std;

typedef device_matrix<float> mat;

enum Method{
	ALL, 
	BATCH, 
	ONE
};

class DNN{
public:
	DNN();
	DNN(Dataset* pData, float learningRate, const vector<size_t>& v, Method method);
//	DNN(const string& fn);
	~DNN();

	void train(size_t batchSize, size_t maxEpoch);
	void predict(vector<size_t>& result, const mat& inputMat);

	size_t getInputDimension();
	size_t getOutputDimension();
	size_t getNumLayers();
	void save(const string& fn);

	void debug();

private:
	void feedForward(mat& ouputMat, const mat& inputMat, bool train);
	void backPropagate(const mat& foutMat, float learningRate);

	Dataset* _pData;
	float _learningRate;
	Method _method;
	vector<Sigmoid*> _transforms;
	vector<float> _validateAccuracy;
};


#endif
