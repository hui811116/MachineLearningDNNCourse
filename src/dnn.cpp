#include "dnn.h"
// undefined yet
#include "dataset.h"
#include <vector>
#include <fstream>

using namespace std;

DNN::DNN(){
}
DNN::DNN(Dataset){
}
DNN::DNN(ifstream& ){
}
DNN::~DNN(){
}

void DNN::train(Dataset input, method type){
}

void DNN::predict(dataset input, vector<float>& result){
}

void DNN::save(ofstream* out){
}

//helper function

bool DNN::feedForward(Dataset input,vector<float>& output){
}
bool DNN::backPropagate(Dataset error){
}
