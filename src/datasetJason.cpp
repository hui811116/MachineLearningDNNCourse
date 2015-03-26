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
	int trainSize = trainProp*getNumOfData();
	int validSize = getNumOfData() - trainProp;
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
	_trainX = new float*[trainSize];
	
}
