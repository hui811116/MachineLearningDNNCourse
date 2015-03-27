#include <iostream>
#include "dataset.h"
#include "dnn.h"
using namespace std;
typedef device_matrix<float> mat;
int main(){
	cout << "This is the data set test\n";	
	size_t  phonemeNum = 39;
	size_t trainDataNum = 20000;
	size_t testDataNum = 1000;
	size_t labelDataNum = 1124823;
	size_t labelNum = 48;

	const char* trainFilename = "/home/larry/Documents/data/MLDS_HW1_RELEASE_v1/mfcc/train.ark";	
	const char* testFilename = "/home/larry/Documents/data/MLDS_HW1_RELEASE_v1/mfcc/test.ark";
	const char* labelFilename = "/home/larry/Documents/data/MLDS_HW1_RELEASE_v1/label/train.lab";
	
	Dataset test = Dataset(trainFilename, trainDataNum, testFilename, testDataNum, labelFilename,labelDataNum, labelNum,phonemeNum);
    // segmentation
	test.dataSegment(0.8);
	// test
	/*
	mat trainBatch, validBatch;
	mat batchLabel;
	//test.getBatch(5, batch, batchLabel);
	vector<size_t> trainPhoneme;
	vector<size_t> validPhoneme;
	test.getTrainSet( 32, trainBatch, trainPhoneme );
	test.getValidSet( validBatch, validPhoneme );
	*/
	cout << "constructing dnn:\n";
	vector<size_t> dimension;
	dimension.push_back(39);
	dimension.push_back(5);
	dimension.push_back(48);
	DNN dnn( &test, 0.1, dimension, BATCH );
	cout << "start dnn training:\n";
	dnn.train(5);
}
