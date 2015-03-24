#ifndef __DATASET_H_
#define __DATASET_H_

#include <string>
#include <vector>
#include <fstream>

using namespace std;

class Dataset{
public:
	Dataset();
	Dataset(const string& fn);
	Dataset(const Dataset& data);
	~Dataset();

	size_t getNumOfData();
	size_t getInputDim();
	size_t getOutputDim();
private:
	size_t _featureDimension;
	size_t _stateDimension;
	size_t _numOfData;
	vector<float>* _dataVectors;
};

#endif 