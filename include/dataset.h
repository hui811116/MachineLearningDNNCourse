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
	// dataset parameters
	size_t _featureDimension;
	size_t _stateDimension;
	size_t _numOfData;
	size_t _numOfPhoneme;
	int    _trainSize;
	int    _validSize;
	// datasetJason private functions
	void   dataSegment( float trainProp);
	mat    outputNumtoBin(int* outputVector); // change 0~47 to a 48 dim mat
	//mat    fltFeatureToMat(float** inputFeature); 
	
	
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
