#include "dataset.h"
#include <algorithm>
#include <vector>
using namespace std;

void Dataset::getBatch(int batchSize, mat batch, mat batchLabel){
	
}

void Dataset::getTrainSet(int trainSize, mat trainData, mat trainLabel){

}

void Dataset::getValidSet(mat validData, mat validLabel){

}

void Dataset::dataSegment( float trainProp ){
	// segment data into training and validating set
	_trainSize = trainProp*getNumOfData();
	_validSize = getNumOfData() - trainProp;
	
	//create random permutation
	vector<int> randIndex;
	
	for (int i = 0; i < getNumOfData(); i++){
		randIndex.push_back( i+1 );
	}
	random_shuffle(randIndex.begin(), randIndex.end());
	// print shuffled data
	for (int i = 0; i < getNumOfData(); i++){
		cout << randIndex[i] <<" ";
	}
	// 
	_trainX = new float*[_trainSize];
	_trainY = new int[_trainSize];
	for (int i = 0; i < _trainSize; i++){
		_trainX[i] = _dataMatrix[ randIndex[i] ]; 
		_trainY[i] = _labelMatrix[ randIndex[i] ];  // depends on ahpan
	}
	_validX = new float*[_validSize];
	_validY = new int[_validSize];
	for (int i = 0; i < _validSize; i++){
		_validX[i] = _dataMatrix [ randIndex[_trainSize + i] ];
		_validY[i] = _labelMatrix[ randIndex[_trainSize + i] ];
	}
	
}
