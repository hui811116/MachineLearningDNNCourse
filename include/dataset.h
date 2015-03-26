#ifndef DATASET_H_
#define DATASET_H_

#include <string>
#include <vector>
#include <fstream>
#include "device_matrix.h"
using namespace std;
typedef device_matrix<float> mat;
class Dataset{
public:
	Dataset();
	Dataset(const char* fn, size_t dataNum, size_t phonemeNum);
	Dataset(const Dataset& data);
	~Dataset();
	
	
	size_t getNumOfData();
	size_t getInputDim();
	size_t getOutputDim();
	void   getBatch(int batchSize, mat batch, mat batchLabel);
	void   getTrainSet(int trainSize, mat trainData, mat trainLabel);
	void   getValidSet(mat validData, mat validLabel);
	unsigned int split(const string &txt, vector<string> &strs, char ch);
private:
	size_t _featureDimension;
	size_t _stateDimension;
	size_t _numOfData;
	size_t _numOfPhoneme;
	void   dataSegment();
	string* _dataNameMatrix;
	float** _dataMatrix;
	//size_t _dataMatrix[][_numOfPhoneme]*;
};

#endif 
