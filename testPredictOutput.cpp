#include <iostream>
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

int main(int argc, char**argv){
	
	size_t labelNum = 48;
	size_t phonemeNum = 39;
	size_t trainDataNum = 5000;
//	size_t trainDataNum = 1124823;
	size_t testDataNum = 180406;
	size_t labelDataNum = 1124823;
	size_t inFtreDim = 69;
	size_t outFtreDim = 48;
	const char* trainFilename = "/home/larry/Documents/data/MLDS_HW1_RELEASE_v1/fbank/train.ark";	
	const char* testFilename = "/home/larry/Documents/data/MLDS_HW1_RELEASE_v1/fbank/test.ark";
	const char* labelFilename = "/home/larry/Documents/data/MLDS_HW1_RELEASE_v1/label/train.lab";
	const char* labelMapFilename = "/home/larry/Documents/data/MLDS_HW1_RELEASE_v1/phones/48_39.map";

		
	Dataset dataset = Dataset(trainFilename, trainDataNum, testFilename, testDataNum, labelFilename, labelDataNum, labelNum, inFtreDim, outFtreDim, phonemeNum);
	dataset.loadTo39PhonemeMap(labelMapFilename);
	//Dataset dataset = Dataset();
	DNN dnn ;
	dnn.load("best0.65.mdl");
	dnn.setDataset(&dataset);	
	dnn.setLearningRate(0.004);
	mat testSet = dataset.getTestSet();
//	cout<<"gg1"<<endl;
	//dataset.getTestSet(testSet);
	vector<size_t> testResult;
	dnn.predict(testResult, testSet);//dataset.getTestSet());
//	cout<<"gg2"<<endl;
	ofstream ofs("testResult.dat");
	if(ofs.is_open()){
		for (size_t i=0;i<testResult.size();i++){
			ofs<<testResult.at(i)<<endl;	
		}
	}
	ofs.close();

	dataset.saveCSV(testResult);	
	return 0;
}
