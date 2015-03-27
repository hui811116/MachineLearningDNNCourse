#include <iostream>
#include "dataset.h"
using namespace std;
typedef device_matrix<float> mat;
int main(){
	cout << "This is the data set test\n";	
	size_t  phonemeNum = 39;
	size_t trainDataNum = 20;
	size_t testDataNum = 10;
	size_t labelDataNum = 1124823;
	size_t labelNum = 48;

	const char* trainFilename = "/home/larry/Documents/data/MLDS_HW1_RELEASE_v1/mfcc/train.ark";	
	const char* testFilename = "/home/larry/Documents/data/MLDS_HW1_RELEASE_v1/mfcc/test.ark";
	const char* labelFilename = "/home/larry/Documents/data/MLDS_HW1_RELEASE_v1/label/train.lab";
	
	Dataset test = Dataset(trainFilename, trainDataNum, testFilename, testDataNum, labelFilename,labelDataNum, labelNum,phonemeNum);
	cout << "construct dataset done\n";
    // segmentation
	test.dataSegment(0.8);
	// test
	mat batch;
	mat batchLabel;
	test.getBatch(5, batch, batchLabel);
}
