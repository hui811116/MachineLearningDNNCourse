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
	void   getBatch(int batchSize, mat& batch, mat& batchLabel);
	void   getTrainSet(int trainSize, mat& trainData, vector<size_t>& trainLabel);
	void   getValidSet(mat& validData, vector<size_t>& validLabel);
	void   dataSegment( float trainProp);
private:
	// dataset parameters
	size_t _featureDimension;
	size_t _stateDimension;
	size_t _numOfData;
	size_t _numOfPhoneme;
	int    _trainSize;
	int    _validSize;
	// datasetJason private functions
	mat    outputNumtoBin(int* outputVector, int vectorSize);
		// change 0~47 to a 48 dim mat
	mat    inputFtreToMat(float** input, int r, int c);	
	
	// original data
	string* _dataNameMatrix; // frame name
	float** _dataMatrix; // input MFCC features
	int* _labelMatrix; // output phoneme changed to integer
	
	// storing training matrix

	float** _trainX;
	float** _validX;
	int* _trainY;
	int* _validY;
};

#endif 
