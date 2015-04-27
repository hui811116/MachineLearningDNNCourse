#ifndef DNN_H
#define DNN_H
#include <vector>
#include <string>
#include <device_matrix.h>
#include "transforms.h"
#include "dataset.h"
using namespace std;

typedef device_matrix<float> mat;

enum Method{
	ALL, 
	BATCH, 
	ONE
};

enum Init{
	UNIFORM,
	NORMAL,
	RBM,
};

class DNN{
public:
	DNN();
	DNN(Dataset* pData, float learningRate,float momentum,float variance,Init init, const vector<size_t>& v, Method method);
	~DNN();

	void train(size_t batchSize, size_t maxEpoch, size_t trainSetNum, size_t validSetNum, float alpha);
	void predict(vector<size_t>& result, const mat& inputMat);

	void setDataset(Dataset* pData);
	void setLearningRate(float learningRate);
	void setMomentum(float momentum);
	size_t getInputDimension();
	size_t getOutputDimension();
	size_t getNumLayers();
	void save(const string& fn);
	bool load(const string& fn);

private:
	void feedForward(mat& ouputMat, const mat& inputMat, bool train);
	void backPropagate(const mat& foutMat, float learningRate, float momentum);

	Dataset* _pData;
	float _learningRate;
	float _momentum;
	Method _method;
	vector<Transforms*> _transforms;
	vector<float> _validateAccuracy;

};


#endif
