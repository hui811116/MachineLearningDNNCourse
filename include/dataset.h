#ifndef DATASET_H_
#define DATASET_H_

#include <string>
#include <vector>
#include <fstream>

using namespace std;
//typedef device_matrix<float> mat;
class Dataset{
public:
	Dataset();
	Dataset(const char* fn, size_t dataNum, size_t phonemeNum);
	Dataset(const Dataset& data);
	~Dataset();
	
	
	size_t getNumOfData();
	size_t getInputDim();
	size_t getOutputDim();
private:
	size_t _featureDimension;
	size_t _stateDimension;
	size_t _numOfData;
	size_t _numOfPhoneme;
	string* _dataNameMatrix;
	//size_t _dataMatrix[][_numOfPhoneme]*;
};

#endif 
