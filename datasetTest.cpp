#include <iostream>
#include <fstream>
#include "dataset.h"

int main()
{
	size_t  phonemeNum = 39;
	size_t trainDataNum = 20;
	size_t testDataNum = 10;
	size_t labelDataNum = 1124823;
	size_t labelNum = 48;
	size_t inputDim = 39;
	size_t outputDim = 48;

	const char* trainFilename = "/home/larry/Documents/data/MLDS_HW1_RELEASE_v1/mfcc/train.ark";	
	const char* testFilename = "/home/larry/Documents/data/MLDS_HW1_RELEASE_v1/mfcc/test.ark";
	const char* labelFilename = "/home/larry/Documents/data/MLDS_HW1_RELEASE_v1/label/train.lab";
	const char* labelMapPath = "/home/larry/Documents/data/MLDS_HW1_RELEASE_v1/phones/48_39.map";	

	Dataset test = Dataset(trainFilename, trainDataNum, testFilename, testDataNum, labelFilename,labelDataNum, labelNum, inputDim, outputDim,phonemeNum);
	test.loadTo39PhonemeMap(labelMapPath);
	test.printTo39PhonemeMap(test.getTo39PhonemeMap());
//	test.printLabelMap(test.getLabelMap());
	return 0;
}
