#include "dataset.h"
#include <iostream>
#include <string>
#include <vector>
//typedef device_matrix<float> mat;

Dataset::Dataset(){
	_featureDimension=0;
	_stateDimension=0;
	_numOfData=0;
	_numOfPhoneme=0;
	_trainSize=0;
	_validSize=0;
}
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
	if(!fin) cout<<"Can't open this file!!!\n";
	string s, tempStr;
	while(getline(fin, s)&&count<dataNum){
		count++;

		cout<<count<<endl;
		unsigned int pos  = s.find(" ");
		unsigned int initialPos = 0;
		split=0;
		while(split<phonemeNum+1){
			dataCount++;
			split++;
			
			tempStr= s.substr(initialPos, pos-initialPos);
			if (split==1){
				*(_dataNameMatrix+count-1) = tempStr;
				
			}
			else{
				_dataMatrix[split-2][count-1] = atof(tempStr.c_str());
			}		
			initialPos = pos+1;
			pos=s.find(" ", initialPos);
		}		
	}		
	cout<<count<<endl;
	cout<<dataCount<<endl;
	
	

};
Dataset::Dataset(const Dataset& data){};
Dataset::~Dataset(){
	
	if(_numOfData!=0)
		delete [] _dataNameMatrix;

	for(int i =0 ; i<_numOfPhoneme;i++)
		delete _dataMatrix[i];
	if(_numOfPhoneme!=0)
	delete [] _dataMatrix;
	//TODO deletion for pointers
	// NOTE:: deletion for _trainX _validX _trainY _validY need careful implementation!!
};

size_t Dataset::getNumOfData(){ return _numOfData; }
size_t Dataset::getInputDim(){}
size_t Dataset::getOutputDim(){}

