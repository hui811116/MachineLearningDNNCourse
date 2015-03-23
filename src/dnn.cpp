#include "dnn.h"
#include "dataset.h"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <device_matrix.h>

#define TEST_SET_PORTION 0.75

using namespace std;

typedef device_matrix<float> mat;
typedef thrust::device_vector<float> vec;

DNN::DNN(){
}
DNN::DNN(Dataset& data){
	size_t numOfTestSet = data.getNumOfData * TEST_SET_PORTION;
	 
}
DNN::DNN(const string& fn){
	ifstream ifs(fn, std::ifstream::in);
	if(!ifs.is_open()){
		cerr << "Cannot open file: " << fn << endl;
		exit(1);
	}
}
DNN::~DNN(){
}

//void DNN::train(Dataset input, method type){
//}

//void DNN::predict(dataset input, vector<float>& result){
//}

void DNN::save(const string& fn){
}

//helper function

bool DNN::feedForward(vector<float>& output){
}
bool DNN::backPropagate(){
}
