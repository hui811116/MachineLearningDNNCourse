#include "dnn.h"
#include "dataset.h"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <device_matrix.h>

using namespace std;

typedef device_matrix<float> mat;

DNN::DNN(){}
DNN::DNN(Dataset& data, float learningRate, const vector<size_t>& v, Method method):_data(data), _learningRate(learningRate), _method(method){
	size_t numOfLayers = v.size();
	for(size_t i = 0; i < numOfLayers-1; i++){
		Sigmoid* pTransform = new Sigmoid(v.at(i), v.at(i+1));
		_transforms.push_back(pTransform);
	}
}
DNN::~DNN(){
	while(!_transforms.empty()){
		delete _transforms.back();
		_transforms.pop_back();
	}
}

void DNN::train(){
}

void DNN::predict(vector<size_t>& result, const mat& inputMat){
	mat outputMat;
	feedForward(outputMat, inputMat);
	result.reserve(outputMat.getCols());
	for(size_t i = 0; i < outputMat.getCols(); i++){
		float tempMaX = outputMat(0, i);
		size_t idx = 0;
		for(size_t j = 0; j < outputMat.getRows(); j++){
			if(tempMax < outputMat(j, i)){
				tempMax = outputMat(j, i);
				idx = j;
			}
		}
		result.push_back(idx);
	}
}

size_t DNN::getInputDimension(){
	return _transforms.front()->getInputDimension();
}

size_t DNN::getOutputDimension(){
	return _transforms.back()->getOutputDimension();
}

size_t DNN::getNumLayers(){
	return _transforms.size()+1;
}

void DNN::save(const string& fn){
}

//helper function

void DNN::feedForward(mat& outputMat, const mat& inputMat){
	mat tempInputMat = inputMat;
	for(size_t i = 0; i < _transforms->size(); i++){
		(_transforms.at(i))->feedForward(outputMat, tempInputMat);
		tempInputMat = outputMat;
	}
}

//The delta of last layer = _sigoutdiff & grad(errorFunc())
void DNN::backPropagate(mat& errorMat, const mat& deltaMat){
	mat tempMat = deltaMat;
	for(int i = _transforms.size()-1; i >= 0; i--){
		(_transforms.at(i))->backPropagate(errorMat, tempMat);
		(_transforms.at(i))->update();
		tempMat = errorMat;
	}
}

//Helper Functions
void

