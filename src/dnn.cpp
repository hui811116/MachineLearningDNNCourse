#include "dnn.h"
#include "dataset.h"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cassert>
#include <device_matrix.h>

#define MAX_EPOCH 10000000

using namespace std;

typedef device_matrix<float> mat;

float computeErrRate(const vector<size_t>& ans, const vector<size_t>& output);
void computeLabel(vector<size_t>& result,const mat& outputMat);

template <typename T>
void randomInit(device_matrix<T>& m) {
	T* h_data = new T [m.size()];
	for (int i=0; i<m.size(); ++i)
		h_data[i] = rand() / (T) RAND_MAX;
	cudaMemcpy(m.getData(), h_data, m.size() * sizeof(T), cudaMemcpyHostToDevice);
	delete [] h_data;
}


DNN::DNN():_pData(NULL), _method(ALL){}

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

void DNN::train(size_t batchSize, size_t maxEpoch = MAX_EPOCH){
	mat trainSet;
	vector<size_t> trainLabel;
	mat validSet;
	vector<size_t> validLabel;
	size_t errRise = 0;
	float Ein = 1;
	float pastEin = Ein;
	float Eout = 1;
	float pastEout = Eout;
	_pData->getTrainSet(25000, trainSet, trainLabel);
	_pData->getValidSet(100000, validSet, validLabel);
	size_t num = 0;
	for(; num < maxEpoch; num++){
		mat batchData;
		mat batchLabel;
		mat batchOutput;
		_pData->getBatch(batchSize, batchData, batchLabel);
		/*
		cout << "Batch Data: " << num << endl;
		batchData.print();
		cout << endl;
		
		cout << "Batch Label: " << num << endl;;
		batchLabel.print();
		cout << endl;
		
		cout << "Transform matrix: " << num << endl;
		for(size_t i = 0; i < _transforms.size(); i++){
			(_transforms.at(i))->print();
			cout << endl;
		}
		*/
		//cout << "start ff\n";
		feedForward(batchOutput, batchData, true);
		//cout << "done feedForward\n";
		float* h_data = new float [batchOutput.size()];
		cudaMemcpy(h_data, batchOutput.getData(), batchOutput.size() * sizeof(float), cudaMemcpyDeviceToHost);
		cout << "segment 1 \n";
		for(size_t j = 0; j < batchOutput.getCols(); j++){
			float sum = 0.0;	
			for(size_t i = 0; i < batchOutput.getRows(); i++){
				sum += h_data[j*batchOutput.getRows() + i];
			}
			for(size_t i = 0; i < batchOutput.getRows(); i++){
				h_data[j*batchOutput.getRows() + i] /= sum;
			}
		}
	
		cudaMemcpy(batchOutput.getData(), h_data, batchOutput.size() * sizeof(float), cudaMemcpyHostToDevice);
		
		delete [] h_data;
		/*
		cout << "Batch output: " << num << endl;
		batchOutput.print();
		cout << endl;
		*/
		mat oneMat(batchOutput.getRows(), batchOutput.getCols(), 1.0);

		//Reserve
		//mat lastDelta;
		//_transforms[_transforms.size()-1]->getSigDiff(lastDelta,(batchOutput-batchLabel) * 2 );
		mat lastDelta(batchOutput & (oneMat-batchOutput) & (batchOutput - batchLabel) * 2);
		backPropagate(lastDelta , _learningRate);

		//backPropagate((batchOutput&(oneMat - batchOutput))&(batchOutput-batchLabel)*(2) , _learningRate);

		vector<size_t> trainResult;
		vector<size_t> validResult;
		predict(trainResult, trainSet);
		predict(validResult, validSet);

		if( num % 200 == 0 ){
			
			//DEBUG
			//	for(size_t t=0;t<trainLabel.size();t++)
			//		cout<<"trainLabel "<<trainLabel[t]<<" trainResult "<<trainResult[t]<<endl;
			//	cout<<endl;
		
			Ein = computeErrRate(trainLabel, trainResult);
			if(Ein > pastEin){
				//cout << "Something wrong had happened, training err does not decrease.\n";
				//exit(1);
			}
			pastEin = Ein;
			Eout = computeErrRate(validLabel, validResult);
			cout.precision(5);
			cout << "Validating error: " << Eout*100 << " %, Training error: " << Ein*100 << " %\n";
			if(Eout > pastEout){
				errRise++;
			}
			else{
				errRise = 0;
			}
			if(errRise == 2){
				break;
			}
		}
		/* save model after a certain steps
		if (num%2000 == 0){
			save("MdlTmp.mdl");
		} 
		*/
	}
	cout << "Finished training for " << num << " epochs.\n";
	//save("debug.mat");	
}

void DNN::predict(vector<size_t>& result, const mat& inputMat){
	mat outputMat(1, 1);
	feedForward(outputMat, inputMat, false);
	float* h_data = new float [outputMat.size()];
	cudaMemcpy(h_data ,outputMat.getData(), outputMat.size() * sizeof(float), cudaMemcpyDeviceToHost);

	for(size_t j = 0; j < outputMat.getCols(); j++){
		float tempMax = h_data[j*outputMat.getRows()];
		size_t idx = 0;		
		for(size_t i = 0; i < outputMat.getRows(); i++){
			if(tempMax < h_data[j*outputMat.getRows() + i]){
				tempMax = h_data[j*outputMat.getRows() + i];
				idx = i;
			}
		}
		result.push_back(idx);
	}
	/*
	for(size_t i = 0; i < outputMat.getRows(); i++){
		for(size_t j = 0; j < outputMat.getCols(); j++){
			cout << h_data[j*outputMat.getRows() + i] << " ";
		}
		cout << endl;
	}
	
	cout << endl;
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
	ofstream ofs(fn);
	if (ofs.is_open()){
		for(size_t i = 0; i < _transforms.size(); i++){
			(_transforms.at(i))->write(ofs);
		}
	}
	ofs.close();
}

void DNN::debug(){

	mat testMat(getInputDimension(), 3);
	mat testLabel(getOutputDimension(),3);
	randomInit(testMat);
	randomInit(testLabel);
	cout.precision(5);
	testMat.print();
//	cout << endl;
//	testLabel.print();
	for(size_t i = 0; i < _transforms.size(); i++){
		(_transforms.at(i))->print();
		cout << endl;
	}
	mat output;
		feedForward(output,testMat,true);
	mat one(output.getRows(),output.getCols(),1.0);
	mat last(output & (one-output) & (output-testLabel) * 2);
	backPropagate(last,_learningRate);
/*
	for(size_t i = 0; i < _transforms.size(); i++){
		(_transforms.at(i))->print();
		cout << endl;
	}
*/
/*
	vector<size_t> result;
	predict(result, testMat);
	cout << "result size:" << result.size() << endl;
	for(size_t i = 0; i < result.size(); i++){
		cout << result.at(i) << endl;
	}
*/
	cout<<"End of debug!"<<endl;
}
//helper function


void DNN::feedForward(mat& outputMat, const mat& inputMat, bool train){
	mat tempInputMat = inputMat;
	for(size_t i = 0; i < _transforms.size(); i++){
		(_transforms.at(i))->forward(outputMat, tempInputMat, train);
		tempInputMat = outputMat;
	}
	//outputMat.print();
}

//The delta of last layer = _sigoutdiff & grad(errorFunc())
void DNN::backPropagate(const mat& deltaMat, float learningRate){
	mat tempMat = deltaMat;
	mat errorMat;
	for(int i = _transforms.size()-1; i >= 0; i--){
		(_transforms.at(i))->backPropagate(errorMat, tempMat, learningRate);
		tempMat = errorMat;
	}
}

//Helper Functions

void computeLabel(vector<size_t>& result,const mat& outputMat){
	float* h_data = new float [outputMat.size()];
	cudaMemcpy(h_data ,outputMat.getData(), outputMat.size() * sizeof(float), cudaMemcpyDeviceToHost);

	for(size_t j = 0; j < outputMat.getCols(); j++){
		float tempMax = h_data[j*outputMat.getRows()];
		size_t idx = 0;		
		for(size_t i = 0; i < outputMat.getRows(); i++){
			if(tempMax < h_data[j*outputMat.getRows() + i]){
				tempMax = h_data[j*outputMat.getRows() + i];
				idx = i;
			}
		}
		result.push_back(idx);
	}
	delete [] h_data;
}

float computeErrRate(const vector<size_t>& ans, const vector<size_t>& output){
	assert(ans.size() == output.size());
	size_t accCount = 0;
	for(size_t i = 0; i < ans.size(); i++){
		if(ans.at(i) == output.at(i)){
			accCount++;
		}
	}
	return 1.0-(float)accCount/(float)ans.size();
}
