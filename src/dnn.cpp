#include "dnn.h"
#include "dataset.h"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <device_matrix.h>

using namespace std;

typedef device_matrix<float> mat;
typedef thrust::device_vector<float> vec;

DNN::DNN(){}
DNN::DNN(Dataset& data, size_t numOfLayer, float learningRate, const vector<size_t>& v, Method method):_data(data), _numOfLayer(numOfLayer), _learningRate(learningRate), _method(method){
	_inputDimension = v.front();
	_outputDimension = v.back();
	_layers = new vector<Sigmoid*>();	
	for(size_t i = 0; i < numOfLayer-1; i++){
		Sigmoid* layer_ptr = new Sigmoid(v.at(i), v.at(i+1));
		_layers.push_back(layer_ptr); 
	}
}
//DNN::DNN(const string& fn){
//	ifstream ifs(fn, std::ifstream::in);
//	if(!ifs.is_open()){
//		cerr << "Cannot open file: " << fn << endl;
//		exit(1);
//	}
//}
DNN::~DNN(){
	while(!_layers->empty()){
		delete _layers->back();
		_layers->pop_back();
	}
	delete _layers;
}

void DNN::train(){
}

vector<float>* DNN::predict(const vector<float>& inputVec){
	vector<float>* temp = &inputVec;
	for(size_t i = 0; i < _layers->size(); i++){
		(_layers->at(i))->feedForward(*temp);
		temp = (_layers->at(i)->getSigOut());
	}
	return temp;
}

void DNN::save(const string& fn){
}

//helper function

bool DNN::feedForward(const vector<float>& inputVec){
i
}
//The delta of last layer = _sigoutdiff & grad(errorFunc())
bool DNN::backPropagate(vector<float>& error){
}
