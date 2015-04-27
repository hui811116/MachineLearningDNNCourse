#include <iostream>
#include <ctime>
#include "dataset.h"
#include <cstdlib>
#include <cstdio>
#include "dnn.h"
using namespace std;
typedef device_matrix<float> mat;
int main(){
	srand(time(NULL));
	cout << "This is the data set test\n";	
	size_t  phonemeNum = 39;
	size_t trainDataNum = 1000;
	size_t testDataNum = 10;
	size_t labelDataNum = 1124823;
	size_t labelNum = 48;

	const char* trainFilename = "/home/larry/Documents/data/MLDS_HW1_RELEASE_v1/mfcc/train.ark";	
	const char* testFilename = "/home/larry/Documents/data/MLDS_HW1_RELEASE_v1/mfcc/test.ark";
	const char* labelFilename = "/home/larry/Documents/data/MLDS_HW1_RELEASE_v1/label/train.lab";
	
	Dataset test = Dataset(trainFilename, trainDataNum, testFilename, testDataNum, labelFilename,labelDataNum, labelNum,phonemeNum);
    // segmentation
	//test.prtPointer(test.getTrainDataMatrix(), phonemeNum,trainDataNum);
	//test.prtPointer(test.getTestDataMatrix(), phonemeNum,testDataNum);
	test.dataSegment(0.8);
	// test
	
	mat trainBatch, validBatch, batch;
	mat batchLabel;
	test.getBatch(5, batch, batchLabel);
	//cout << "print Label:\n";
	//batchLabel.print();
	//vector<size_t> trainPhoneme;
	//vector<size_t> validPhoneme;
	//test.getTrainSet( 32, trainBatch, trainPhoneme );
	
	//cout << "output Valid phoneme:\n";
	vector<size_t> validPhoneme;
	test.getValidSet( validBatch, validPhoneme );
	//for (int i = 0; i < validPhoneme.size(); i++)
	//	cout <<validPhoneme[i] << " ";

	cout << "constructing dnn:\n";
	vector<size_t> dimension;
	dimension.push_back(39);
	dimension.push_back(5);
	dimension.push_back(48);
	DNN dnn( &test, 0.01, dimension, BATCH );
	cout << "start dnn training:\n";
	dnn.train(5, 100000);
}
