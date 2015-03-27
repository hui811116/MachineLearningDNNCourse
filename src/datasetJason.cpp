#include "dataset.h"
#include <algorithm>
#include <vector>
#include <cstdlib> // rand()
using namespace std;
typedef device_matrix<float> mat;
void Dataset::getBatch(int batchSize, mat& batch, mat& batchLabel){
	// random initialize indices for this batch
	int* randIndex = new int [batchSize];
	for (int i = 0; i < batchSize; i++){
		randIndex[i] = rand() % _trainSize; // shall check RAND_MAX 
	}
	float** batchFtre = new float*[batchSize];
	int*    batchOutput = new int[batchSize];
	for (int i = 0; i < batchSize; i++){
		batchFtre[i] = _trainX[ randIndex[i] ];
		batchOutput[i] = _trainY[ randIndex[i] ];
	}
	// convert them into mat format
	batch = inputFtreToMat( batchFtre, _numOfPhoneme, batchSize);
	batchLabel = outputNumtoBin( batchOutput, batchSize );

	// for debugging, print both matrices
	cout << "This is the feature matrix\n";
	batch.print();
	cout << "This is the label matrix\n";
	batchLabel.print();
}

void Dataset::getTrainSet(int trainSize, mat& trainData, vector<size_t>& trainLabel){

}

void Dataset::getValidSet(mat& validData, vector<size_t>& validLabel){

}

void Dataset::dataSegment( float trainProp ){
	cout << "start data segmenting:\n";
	cout << "num of data is "<< _numOfData << endl;
	// segment data into training and validating set
	_trainSize = trainProp*getNumOfData();
	_validSize = getNumOfData() - _trainSize;
	
	//create random permutation
	vector<int> randIndex;
	
	for (int i = 0; i < getNumOfData(); i++){
		randIndex.push_back( i );
	}
	random_shuffle(randIndex.begin(), randIndex.end());
	// print shuffled data
	cout << "start shuffling:\n";
	for (int i = 0; i < getNumOfData(); i++){
		cout << randIndex[i] <<" ";
	}
	// 
	cout << "put feature into training set\n";
	cout << "trainingsize = " << _trainSize <<endl;
	_trainX = new float*[_trainSize];
	_trainY = new int[_trainSize];
	for (int i = 0; i < _trainSize; i++){
		_trainX[i] = _dataMatrix[ randIndex[i] ]; 
		_trainY[i] = _labelMatrix[ randIndex[i] ];  // depends on ahpan
	}
	cout << "put feature into validating set\n";
	cout << "validatingsize = " << _validSize <<endl;
	_validX = new float*[_validSize];
	_validY = new int[_validSize];
	for (int i = 0; i < _validSize; i++){
		_validX[i] = _dataMatrix [ randIndex[_trainSize + i] ];
		_validY[i] = _labelMatrix[ randIndex[_trainSize + i] ];
	}
	
}
mat Dataset::outputNumtoBin(int* outputVector, int vectorSize)
{
	float* tmpVector = new float[ vectorSize * _numOfPhoneme ];
	for (int i = 0; i < vectorSize; i++){
		for (int j = 0; j < _numOfPhoneme; j++){
			*(tmpVector + i*_numOfPhoneme + j) = (outputVector[i] == j)?1:0;
		}
	}

	mat outputMat(tmpVector, _numOfPhoneme, vectorSize);
	delete[] tmpVector;
	return outputMat;
}
mat Dataset::inputFtreToMat(float** input, int r, int c){
	// r shall be the number of phonemes
	// c shall be the number of data
	float* inputReshaped = new float[r * c];
	for (int i = 0; i < c; i++){
		for (int j = 0; j < r; j++){
			*(inputReshaped + i*r + j) = *(*(input + i) +j);
		}
	}
	mat outputMat(inputReshaped, r, c);
	delete[] inputReshaped;
	return outputMat;
}
