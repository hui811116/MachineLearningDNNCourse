#include "dnn.h"
// undefined yet
#include "dataset.h"
#include <vector>
#include <fstream>

using namespace std;

dnn::dnn(){
}
dnn::dnn(dataset){
}
dnn::dnn(ifstream*){
}
dnn::~dnn(){
}

void dnn::train(dataset _input,method _type){
}

void dnn::predict(dataset _input,vector<float>& _result){
}

void dnn::save(ofstream* _out){
}

//helper function

bool dnn::feedForward(dataset _input,vector<float>& _output){
}
bool dnn::backPropagate(dataset _error){
}
