#include <iostream>
#include "dnn.h"
#include "dataset.h"

using namespace std;



typedef device_matrix<float> mat;


int main(int argc, char**argv){
	
	size_t labelNum = 48;
	size_t phonemeNum = 39;
	size_t trainDataNum = 5000;
//	size_t trainDataNum = 1124823;

	size_t testDataNum = 180406;
	size_t testDataNum1 = 100000;
	size_t testDataNum2 = 80406;//180406;
	size_t labelDataNum = 1124823;
	size_t inFtreDim = 69;
	size_t outFtreDim = 48;
	const char* trainFilename = "/home/larry/Documents/data/MLDS_HW1_RELEASE_v1/fbank/train.ark";	
//	const char* testFilename = "/home/larry/Documents/data/MLDS_HW1_RELEASE_v1/fbank/test.ark";
	
	const char* testFilename1 = "test1.ark";
	const char* testFilename2 = "test2.ark";
	const char* testFilename = "test.ark";
	const char* labelFilename = "/home/larry/Documents/data/MLDS_HW1_RELEASE_v1/label/train.lab";
	const char* labelMapFilename = "/home/larry/Documents/data/MLDS_HW1_RELEASE_v1/phones/48_39.map";
	
		
	Dataset dataset = Dataset(trainFilename, trainDataNum, testFilename1, testDataNum1, labelFilename, labelDataNum, labelNum, inFtreDim, outFtreDim, phonemeNum);
	dataset.loadTo39PhonemeMap(labelMapFilename);
	//Dataset dataset = Dataset();
	DNN dnn ;
	dnn.load("model/best0.55.mdl");
	dnn.setDataset(&dataset);	
	dnn.setLearningRate(0.004);
	mat testSet = dataset.getTestSet();
//	cout<<"gg1"<<endl;
	//dataset.getTestSet(testSet);
	vector<size_t> testResult;
	dnn.predict(testResult, testSet);//dataset.getTestSet());
//	cout<<"gg2"<<endl;
	cout<<"testResult size: "<<testResult.size()<<endl;
	ofstream ofs("testResult2.dat");
	if(ofs.is_open()){
		for (size_t i=0;i<testResult.size();i++){
			ofs<<testResult.at(i)<<endl;	
		}
	}
	ofs.close();

	Dataset dataset2 = Dataset(trainFilename, trainDataNum, testFilename2, testDataNum2, labelFilename, labelDataNum, labelNum, inFtreDim, outFtreDim, phonemeNum);
	dataset2.loadTo39PhonemeMap(labelMapFilename);
	//Dataset dataset = Dataset();
	DNN dnn2 ;
	dnn2.load("model/best0.55.mdl");
	dnn2.setDataset(&dataset2);	
	dnn2.setLearningRate(0.004);
	mat testSet2 = dataset2.getTestSet();
//	cout<<"gg1"<<endl;
	//dataset.getTestSet(testSet);
	vector<size_t> testResult2;
	dnn2.predict(testResult2, testSet2);//dataset.getTestSet());
	cout<<"testResult2 size: "<<testResult2.size()<<endl;
	testResult2.insert(testResult2.begin(), testResult.begin(), testResult.end());
	
	cout<<"testResultMerge size: "<<testResult2.size()<<endl;
	Dataset dataset3 = Dataset(trainFilename, trainDataNum, testFilename, testDataNum, labelFilename, labelDataNum, labelNum, inFtreDim, outFtreDim, phonemeNum);
	
	dataset3.loadTo39PhonemeMap(labelMapFilename);
	dataset3.saveCSV(testResult2);	

	return 0;
}
