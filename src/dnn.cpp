#include "dnn.h"
// undefined yet
//#include "dataset.h"
#include <vector>
#include <string>
#include <fstream>

using namespace std;

DNN::DNN(){
}
//DNN::DNN(Dataset& data){
}
DNN::DNN(const string& fn){
	ifstream ifs(fn);
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
