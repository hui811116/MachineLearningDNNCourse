#include <iostream>
#include <string>
#include <vector>
#include <ctime>
#include <device_matrix.h>
#include "dnn.h"
#include "dataset.h"

using namespace std;

typedef device_matrix<float> mat;

template <typename T>
void randomInit(device_matrix<T>& m) {
	T* h_data = new T [m.size()];
	for (int i=0; i<m.size(); ++i)
		h_data[i] = rand() / (T) RAND_MAX;
	cudaMemcpy(m.getData(), h_data, m.size() * sizeof(T), cudaMemcpyHostToDevice);
	delete [] h_data;
}

int main(int argc, char** argv){
	srand((unsigned)time(0));
	
	size_t labelNum = 48;
	size_t phonemeNum = 39;
	size_t trainDataNum = 1124823;
	size_t testDataNum = 180406;
	size_t labelDataNum = 1124823;

	const char* trainFilename = "/home/larry/Documents/data/MLDS_HW1_RELEASE_v1/mfcc/train.ark";	
	const char* testFilename = "/home/larry/Documents/data/MLDS_HW1_RELEASE_v1/mfcc/test.ark";
	const char* labelFilename = "/home/larry/Documents/data/MLDS_HW1_RELEASE_v1/label/train.lab";
		
	Dataset dataset = Dataset(trainFilename, trainDataNum, testFilename, testDataNum, labelFilename, labelDataNum, labelNum, phonemeNum);

	// set network structure
	vector<size_t> dimensions;
	dimensions.push_back(39);
	dimensions.push_back(128);
	dimensions.push_back(48);

	dataset.dataSegment(0.8);
	//start training
	DNN dnn(&dataset, 0.004, dimensions, BATCH);

	//dnn.train(5, 2000);
	//dnn.debug();
	
	dnn.train(500, 15000);
	//dnn.train(200, 10000000);
	dnn.save("MdlEta1e-4.mdl");
	cout << "learnRate: 0.0001 done";
	/*
	DNN dnn2(&dataset, 0.1, dimensions, BATCH);
	//dnn.debug();
	dnn2.train(500, 10000);
	dnn2.save("MdlEta1e-1");
	cout << "learnRate: 0.1 done";

	DNN dnn3(&dataset, 1, dimensions, BATCH);
	//dnn.debug();
	dnn3.train(500, 10000);
	dnn3.save("MdlEta1e-0");
	cout << "learnRate: 1 done";
	*/
	return 0;
}

