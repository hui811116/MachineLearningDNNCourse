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
	mat outputMat(1, 1);
	feedForward(outputMat, inputMat, false);
	cout << endl;
	float* h_data = new float [outputMat.size()];
	cudaMemcpy(h_data ,outputMat.getData(), outputMat.size() * sizeof(float), cudaMemcpyDeviceToHost);

	for(size_t j = 0; j < outputMat.getCols(); j++){
		float tempMax = h_data[j*outputMat.getRows()];
		size_t idx = 0;		
		for(size_t i = 0; i < outputMat.getRows(); i++){
			cout << h_data[j*outputMat.getRows() + i] << " ";
			if(tempMax < h_data[j*outputMat.getRows() + i]){
				tempMax = h_data[j*outputMat.getRows() + i];
				idx = i;
			}
		}
		cout << endl;
		result.push_back(idx);
	}
	/*
	for(size_t i = 0; i < outputMat.getRows(); i++){
		cout << h_data[i] << " ";
		for(size_t j = 1; j < outputMat.getCols(); j++){
			cout << h_data[j*outputMat.getRows() + i] << " ";
		}
		cout << endl;
	}
	
	cout << endl;

	for(size_t j = 0; j < outputMat.getCols(); j++){
		for(size_t i = 0; i < outputMat.getRows(); i++){
			cout << h_data[j*outputMat.getRows() + i] << " ";
		}
		cout << endl;
	}
	*/
	
	delete [] h_data;
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
	mat testMat(getInputDimension(), 3);
	randomInit(testMat);
	testMat.print();
	cout << endl;
	for(size_t i = 0; i < _transforms.size(); i++){
		(_transforms.at(i))->print();
		cout << endl;
	}
	vector<size_t> result;
	predict(result, testMat);
	cout << "result size:" << result.size() << endl;
	for(size_t i = 0; i < result.size(); i++){
		cout << result.at(i) << endl;
	}
}
//helper function

void DNN::feedForward(mat& outputMat, const mat& inputMat, bool train){
	mat tempInputMat = inputMat;
	for(size_t i = 0; i < _transforms.size(); i++){
		(_transforms.at(i))->forward(outputMat, tempInputMat, train);
		tempInputMat = outputMat;
	}
	cout << "finished feedforward!" << endl;
	outputMat.print();
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

