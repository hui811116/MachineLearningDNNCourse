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
		randIndex[i] = rand() % _trainSize; 
	}
	float** batchFtre = new float*[batchSize];
	int*    batchOutput = new int[batchSize];
	for (int i = 0; i < batchSize; i++){
		batchFtre[i] = _trainX[ randIndex[i] ];
		batchOutput[i] = _trainY[ randIndex[i] ];
	}
	// convert them into mat format
	batch = inputFtreToMat( batchFtre, _numOfLabel, batchSize);
	batchLabel = outputNumtoBin( batchOutput, batchSize );

	// for debugging, print both matrices
	/*
	cout << "This is the feature matrix\n";
	batch.print();
	cout << "from trainX pointer:\n";
	prtPointer(batchFtre, _numOfLabel, batchSize);
	cout << "This is the label matrix\n";
	batchLabel.print();
	*/
}

void Dataset::getTrainSet(int trainSize, mat& trainData, vector<size_t>& trainLabel){
	trainData = inputFtreToMat(_trainX, _numOfLabel, trainSize);
	trainLabel.clear();
	for (int i = 0; i < trainSize; i++)
		trainLabel.push_back( _trainY[i] );
	cout << "get Train Set:\n";
	//trainData.print();
}

void Dataset::getValidSet(mat& validData, vector<size_t>& validLabel){
	validData = inputFtreToMat(_validX, _numOfLabel, _validSize);
	validLabel.clear();
	for (int i = 0; i < _validSize; i++)
		validLabel.push_back( _validY[i] );
	cout << "getValidSet:\n";
	validData.print();
}


void Dataset::dataSegment( float trainProp ){
	cout << "start data segmenting:\n";
	cout << "num of data is "<< getNumOfTrainData() << endl;
	// segment data into training and validating set
	_trainSize = trainProp*getNumOfTrainData();
	_validSize = getNumOfTrainData() - _trainSize;
	
	//create random permutation
	vector<int> randIndex;
	
	for (int i = 0; i < getNumOfTrainData(); i++){
		randIndex.push_back( i );
	}
	//random_shuffle(randIndex.begin(), randIndex.end());
	// print shuffled data
	cout << "start shuffling:\n";
	for (int i = 0; i < getNumOfTrainData(); i++){
		cout << randIndex[i] <<" ";
	}
	// 
	cout << "put feature into training set\n";
	cout << "trainingsize = " << _trainSize <<endl;
	_trainX = new float*[_trainSize];
	_trainY = new int[_trainSize];
	for (int i = 0; i < _trainSize; i++){
		_trainX[i] = _trainDataMatrix[ randIndex[i] ]; 
		_trainY[i] = _labelMatrix[ randIndex[i] ];  // depends on ahpan
	}
	cout << "put feature into validating set\n";
	cout << "validatingsize = " << _validSize <<endl;
	_validX = new float*[_validSize];
	_validY = new int[_validSize];
	for (int i = 0; i < _validSize; i++){
		_validX[i] = _trainDataMatrix [ randIndex[_trainSize + i] ];
		_validY[i] = _labelMatrix[ randIndex[_trainSize + i] ];
	}
	// debugging, print out train x y valid x y
	prtPointer(_trainX, _numOfLabel, _trainSize);
	prtPointer(_validX, _numOfLabel, _validSize);
	cout << "print train phoneme:\n";
	for (int i = 0; i < _trainSize; i++)
		cout << _trainY[i] << " ";
	cout << "print valid phoneme:\n";
	for (int i = 0; i < _validSize; i++)
		cout << _validY[i] << " ";
}
mat Dataset::outputNumtoBin(int* outputVector, int vectorSize)
{
	float* tmpVector = new float[ vectorSize * _numOfLabel ];
	for (int i = 0; i < vectorSize; i++){
		for (int j = 0; j < _numOfLabel; j++){
			*(tmpVector + i*_numOfLabel + j) = (outputVector[i] == j)?1:0;
		}
	}

	mat outputMat(tmpVector, _numOfLabel, vectorSize);
	delete[] tmpVector;
	return outputMat;
}
mat Dataset::inputFtreToMat(float** input, int r, int c){
	// r shall be the number of Labels
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
void Dataset::prtPointer(float** input, int r, int c){
	cout << "this prints the pointer of size: " << r << " " << c << endl;
	for (int i = 0; i < c; i++){
		cout << i << endl;
		for(int j = 0; j < r; j++){
			cout <<input[i][j]<<" ";
			if ((j+1)%5 == 0) cout <<endl;
		}
		cout <<endl;
	}
	return;
}
