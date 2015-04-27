#include <iostream>
#include "dataset.h"
#include "dnn.h"
using namespace std;
typedef device_matrix<float> mat;

DNN nnTrain( Dataset* dataPtr, 
             vector<float> learnRate, int xFold, 
			 float trainProp, vector<size_t> dimension){
	// this function outputs the best DNN model
	if ( xfold == 1 ){ // no cross-validation
		dataPtr->dataSegment(trainProp);

	}

}
