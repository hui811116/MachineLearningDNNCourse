#include "dataset.h"
#include <iostream>
#include <string>
#include <vector>
//typedef device_matrix<float> mat;

Dataset::Dataset(){};
Dataset::Dataset(const char* fn, size_t dataNum, size_t phonemeNum){
	_numOfData = dataNum;
	_numOfPhoneme = phonemeNum;

	size_t count  = 0, dataCount = 0;
	short split = 0;	
	_dataNameMatrix  = new string[dataNum];	
	_dataMatrix = new float*[phonemeNum];
	for(int i = 0;i<phonemeNum;i++){
		_dataMatrix[i] = new float [dataNum];
	}
	

	ifstream fin(fn);
	if(!fin) cout<<"Cant't open this file!!!\n";
//	char line[30];
	string s, tempStr;
	while(getline(fin, s)){
		count++;

		cout<<count<<endl;
		unsigned int pos  = s.find(" ");
		unsigned int initialPos = 0;
		split=0;
		while(split<phonemeNum+1){
			//cout<<
			dataCount++;
			split++;
			//cout<<dataCount<<endl;
			//cout<<
			tempStr= s.substr(initialPos, pos-initialPos);
			if (split==1){
				*(_dataNameMatrix+count-1) = tempStr;
			//	dataNum++;
				
			}
			else{
				cout<<"ya1";
				_dataMatrix[split-1][dataCount] = atof(tempStr.c_str());
				cout<<"ya2";
			}		
			//s.substr(initialPos, pos-initialPos);
			//<<endl;
			initialPos = pos+1;
			pos=s.find(" ", initialPos);
		}		
//		if(count==1){
		//cout<<s<<endl;	
//		dataCount++;
	}		
	cout<<count<<endl;
	cout<<dataCount<<endl;
	
	

};
Dataset::Dataset(const Dataset& data){};
Dataset::~Dataset(){
	delete _dataNameMatrix;
	
};

size_t Dataset::getNumOfData(){}
size_t Dataset::getInputDim(){}
size_t Dataset::getOutputDim(){}

