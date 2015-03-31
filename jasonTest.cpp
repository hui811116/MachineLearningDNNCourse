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
	size_t inFtreDim = 39;
	size_t outFtreDim = 48;
	const char* trainFilename = "/home/larry/Documents/data/MLDS_HW1_RELEASE_v1/mfcc/train.ark";	
	const char* testFilename = "/home/larry/Documents/data/MLDS_HW1_RELEASE_v1/mfcc/test.ark";
	const char* labelFilename = "/home/larry/Documents/data/MLDS_HW1_RELEASE_v1/label/train.lab";
	
	Dataset test = Dataset(trainFilename, trainDataNum, testFilename, testDataNum, labelFilename,labelDataNum, labelNum, inFtreDim, outFtreDim, phonemeNum);
    // segmentation
	test.dataSegment(0.8);
	
	
	
	cout << "constructing dnn:\n";
	vector<size_t> dimension;
	dimension.push_back(inFtreDim);
	dimension.push_back(128);
	dimension.push_back(outFtreDim);
	DNN dnn( &test, 0.001, dimension, BATCH );
	//cout << "start dnn training:\n";
	//dnn.train(128, 10000);
	

	cout << "loading existing Mdl:\n";
	dnn.load("MdlEta1e-4.mdl");
	cout << "load done\n";
	cout << "test Data has " << test.getNumOfTestData() << endl;
	mat testData;
	testData = test.getTestSet();
	vector<size_t> result;
	cout << "start predicting:\n";
	dnn.predict(result, testData);
	cout << "save output:\n";
	test.saveCSV(result);
}
