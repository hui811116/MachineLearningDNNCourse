#include "dataset.h"
//typedef device_matrix<float> mat;

Dataset::Dataset(){};
Dataset::Dataset(const char* fn, size_t dataNum, size_t phonemeNum){
	_numOfData = dataNum;
	_numOfPhoneme = phonemeNum;

	size_t count  = 0, dataCount = 0;
	
	ifstream fin(fn);
	if(!fin) cout<<"Cant't open this file!!!\n";
	char line[30];
	while(fin.getline(line, sizeof(line), ' ')){
		count++;
		if(count==1){
			dataCount ++;	
		}
		if(count==40) count=0;
	}

	
	

};
Dataset::Dataset(const Dataset& data){};
Dataset::~Dataset(){};

