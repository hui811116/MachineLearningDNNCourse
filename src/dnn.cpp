#include "dnn.h"
#include "dataset.h"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <device_matrix.h>

using namespace std;

typedef device_matrix<float> mat;

template <typename T>
void randomInit(device_matrix<T>& m) {
  T* h_data = new T [m.size()];
  for (int i=0; i<m.size(); ++i)
    h_data[i] = rand() / (T) RAND_MAX;
  cudaMemcpy(m.getData(), h_data, m.size() * sizeof(T), cudaMemcpyHostToDevice);
  delete [] h_data;
}


DNN::DNN(){}
DNN::DNN(Dataset* pData, float learningRate, const vector<size_t>& v, Method method):_pData(pData), _learningRate(learningRate), _method(method){
	size_t numOfLayers = v.size();
	for(size_t i = 0; i < numOfLayers-1; i++){
		Sigmoid* pTransform = new Sigmoid(v.at(i+1), v.at(i));
		_transforms.push_back(pTransform);
	}
}
DNN::~DNN(){
	while(!_transforms.empty()){
		delete _transforms.back();
		_transforms.pop_back();
	}
}

void DNN::train(size_t batchSize){
}

void DNN::predict(vector<size_t>& result, const mat& inputMat){
	mat outputMat;
	feedForward(outputMat, inputMat, false);
	float* outputData = outputMat.getData();
	for(size_t i = 0; i < outputMat.getCols(); i++){
		float tempMax = outputData[0 + i];
		size_t idx = 0;
		for(size_t j = 0; j < outputMat.getRows(); j++){
			if(tempMax < outputData[j*outputMat.getRows()+i]){
				tempMax = outputData[j*outputMat.getRows()+i];
				idx = j;
			}
		}
		result.push_back(idx);
	}
}

size_t DNN::getInputDimension(){
	return _transforms.front()->getInputDim();
}

size_t DNN::getOutputDimension(){
	return _transforms.back()->getOutputDim();
}

size_t DNN::getNumLayers(){
	return _transforms.size()+1;
}

void DNN::save(const string& fn){
}

void DNN::debug(){
	mat testMat(getInputDimension(), 1);
	randomInit(testMat);
	testMat.print();
	cout << endl;
	for(size_t i = 0; i < _transforms.size(); i++){
		(_transforms.at(i))->print();
		cout << endl;
	}
//	vector<size_t> result;
	mat outputMat;
	feedForward(outputMat, testMat, false);
//  cout << "result size:" << result.size();
//	for(size_t i = 0; i < result.size(); i++){
//		cout << result.at(i) << endl;
//	}
}
//helper function

void DNN::feedForward(mat& outputMat, const mat& inputMat, bool train){
	mat tempInputMat = inputMat;
	for(size_t i = 0; i < _transforms.size(); i++){
		cout << i << endl;
		(_transforms.at(i))->forward(outputMat, tempInputMat, train);
		tempInputMat = outputMat;
	}
	cout << "finish forward algorithm." << endl;
}

//The delta of last layer = _sigoutdiff & grad(errorFunc())
void DNN::backPropagate(mat& errorMat, const mat& deltaMat, float learningRate){
	mat tempMat = deltaMat;
	for(int i = _transforms.size()-1; i >= 0; i--){
		(_transforms.at(i))->backPropagate(errorMat, tempMat, learningRate);
		tempMat = errorMat;
	}
}

//Helper Functions

