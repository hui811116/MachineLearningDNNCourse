#include "dataset.h"
#include <iostream>
#include <string>
#include <vector>
#include <map>
//typedef device_matrix<float> mat;
struct mapData {
	size_t phoneme;
	float* inputFeature;
};
typedef map<string, mapData> NameFtreMap;

Dataset::Dataset(){
	_featureDimension=39;
	_stateDimension=0;
	_numOfTrainData=0;
	_numOfPhoneme=0;
	_trainSize=0;
	_validSize=0;
	
	_trainDataNameMatrix = NULL;
	_testDataNameMatrix = NULL;
	_trainDataMatrix = NULL;
	_testDataMatrix = NULL;
	_labelMatrix = NULL;
	_trainX = NULL;
	_validX = NULL;
	_trainY = NULL;
	_validY = NULL;
}
Dataset::Dataset(const char* trainPath, size_t trainDataNum, const char* testPath, size_t testDataNum, const char* labelPath, size_t labelDataNum, size_t labelNum, size_t inputDim, size_t outputDim, size_t phonemeNum){
	_numOfTrainData = trainDataNum;
	_numOfTestData = testDataNum;
	_numOfLabel = labelNum;
	_numOfPhoneme = phonemeNum;
	_featureDimension = inputDim;
	_stateDimension = outputDim;

	size_t count  = 0, dataCount = 0;
	short split = 0;	
	_trainDataNameMatrix  = new string[trainDataNum];	
	_trainDataMatrix = new float*[trainDataNum];
	for(int i = 0;i<trainDataNum;i++){
		_trainDataMatrix[i] = new float [phonemeNum];
	}
	cout << "inputting train feature file:\n";	
	NameFtreMap InputMap;
	ifstream fin(trainPath);
	if(!fin) cout<<"Can't open this file!!!\n";
	string s, tempStr;
	while(getline(fin, s) && count<trainDataNum){
		count++;

//		cout<<count<<endl;
		unsigned int pos  = s.find(" ");
		unsigned int initialPos = 0;
		split=0;
		string tmpName;
	    mapData tmpData;
		tmpData.inputFeature = new float[inputDim];	
		while(split<inputDim+1){
			dataCount++;
			split++;
			
			tempStr= s.substr(initialPos, pos-initialPos);
			if (split==1){
				//*(_trainDataNameMatrix+count-1) = tempStr;
				//cout<<*(_trainDataNameMatrix+count-1)<<endl;	
				tmpName = tempStr;
			}

			else{
				//_trainDataMatrix[count-1][split-2] = atof(tempStr.c_str());
				//cout<<_trainDataMatrix[count-1][split-2]<<endl;
				tmpData.inputFeature[split-2] = atof(tempStr.c_str());
			}		
			initialPos = pos+1;
			pos=s.find(" ", initialPos);
		}		
		InputMap.insert(pair<string, mapData>(tmpName, tmpData));
	}		
	//cout<<count<<endl;
	//cout<<dataCount<<endl;
	
	fin.close();	
	cout << "inputting testing file:\n";
	size_t testCount = 0, testDataCount = 0;
	 split = 0;	
	_testDataNameMatrix  = new string[testDataNum];	
	_testDataMatrix = new float*[testDataNum];
	for(int i = 0;i<testDataNum;i++){
		_testDataMatrix[i] = new float [inputDim];
	}
	
	//cout<<"Test starts"<<endl;
	ifstream finTest(testPath);
	if(!finTest) cout<<"Can't open this file!!!\n";
	string sTest, tempStrTest;
	while(getline(finTest, sTest)&&testCount<testDataNum){
		testCount++;
		//cout<<testCount<<endl;
		//cout<<count<<endl;
		unsigned int posTest  = sTest.find(" ");
		unsigned int initialPos = 0;
		split=0;
		while(split<inputDim+1){
			testDataCount++;
			split++;
			
			tempStrTest= sTest.substr(initialPos, posTest-initialPos);
			//cout<<"After test subsrt"<<endl;
			if (split==1){
				*(_testDataNameMatrix+testCount-1) = tempStrTest;
				//cout<<*(_testDataNameMatrix+testCount-1)<<endl;	
			}
			else{
				_testDataMatrix[testCount-1][split-2] = atof(tempStrTest.c_str());
				//cout<<_dataMatrix[count-1][split-2]<<endl;
			}		
			initialPos = posTest+1;
			posTest=sTest.find(" ", initialPos);
		}		
	}		
	//cout<<testCount<<endl;
	//cout<<testDataCount<<endl;
	
	finTest.close();

	cout << "inputting training label file:\n";
	size_t countLabel  = 0, labelDataCount = 0, numForLabel=0;
	 split = 0;	
		
	_labelMatrix = new int[labelDataNum]; 

	ifstream finLabel(labelPath);
	if(!finLabel) cout<<"Can't open this file!!!\n";
	string sLabel, tempStrLabel, preLabel= "" ;
	while(getline(finLabel, sLabel)){
		countLabel++;

		//cout<<count<<endl;
		unsigned int pos  = sLabel.find(",");
		unsigned int initialPos = 0;
		split=0;
		string tmpName;
		while(split<2){
			labelDataCount++;
			split++;
			
			tempStrLabel = sLabel.substr(initialPos, pos-initialPos);
			if (split == 1) tmpName = tempStrLabel;

			if (split==2){
				if(tempStrLabel.compare(preLabel)!=0){
					if(_labelMap.find(tempStrLabel)==_labelMap.end()){
						numForLabel++;
						//preLabel = tempStrLabel;
						_labelMap.insert(pair<string, int>(tempStrLabel, numForLabel));	
						//cout<<numForLabel<<endl;
					}
					preLabel = tempStrLabel;
				}

			
			//*(_labelMatrix+countLabel-1)=labelMap.find(tempStrLabel)->second;
			InputMap.find(tmpName)->second.phoneme =_labelMap.find(tempStrLabel)->second; 
			//cout<<tempStrLabel<<endl;		

					
			//*(_labelMatrix+countLabel-1) = tempStrLabel;
			//	cout<<*(_labelMatrix+count-1)<<endl;	
			}
			initialPos = pos+1;
			pos=sLabel.find(",", initialPos);
		}		
	}		
	//cout<<countLabel<<endl;
	//cout<<labelDataCount<<endl;
	
	finLabel.close();	

	// put things into dataMatrix
	cout << "putting them into pointers:\n";
	int mapCount = 0;
	for (NameFtreMap::const_iterator it = InputMap.begin();
		 it != InputMap.end(); ++it){
		 _trainDataMatrix[mapCount] = it->second.inputFeature;
		 _labelMatrix[mapCount] = it->second.phoneme;
		 mapCount ++;
	}
	/*
	for (NameFtreMap::const_iterator it = InputMap.begin();
		 it != InputMap.end(); ++it){
		 delete[] it->second.inputFeature;
	}
	*/
};
Dataset::Dataset(const Dataset& data){};
Dataset::~Dataset(){
	if(_numOfTrainData!=0)
		delete [] _trainDataNameMatrix;

	for(int i =0 ; i<_numOfTrainData;i++)
		delete[] _trainDataMatrix[i];
	if(_numOfPhoneme!=0)
		delete [] _trainDataMatrix;
	
	if(_testDataMatrix != NULL){
		for (int i = 0;i<_numOfTestData;i++){
			delete[] _testDataMatrix[i];
		}
	}
	if(_testDataMatrix != NULL){
		delete []_testDataMatrix;
	}
	if(_testDataNameMatrix != NULL){
		delete []_testDataNameMatrix;
	}
	if(_labelMatrix != NULL){
		delete []_labelMatrix;
	}
	if (_trainX != NULL){
		//for (int i = 0; i < _trainSize; i++ )
		//	delete[] _trainX[i];
		delete[] _trainX;
	}
	if (_validX != NULL){
		//for (int i = 0; i < _validSize; i++ )
		//	delete[] _validX[i];
		delete[] _validX;
	}
	delete[] _trainY;
	delete[] _validY;

	//TODO deletion for pointers
	// NOTE:: deletion for _trainX _validX _trainY _validY need careful implementation!!
};

void Dataset::saveCSV(){
	
	string name, phoneme;
	ofstream fout("Prediction.csv");
	if(!fout){
		cout<<"Can't write the file!"<<endl;
	}
	fout<<"Id,Prediction\n";
	
	fout.close();
}


//Get function
size_t Dataset::getNumOfTrainData(){ return _numOfTrainData; }
size_t Dataset::getNumOfTestData(){return _numOfTestData;}
size_t Dataset::getNumOfLabel(){return _numOfLabel;}
size_t Dataset::getNumOfPhoneme(){return _numOfPhoneme;}
size_t Dataset::getInputDim(){ return _featureDimension; }
size_t Dataset::getOutputDim(){return _stateDimension;}
float** Dataset::getTrainDataMatrix(){return _trainDataMatrix;}
float** Dataset::getTestDataMatrix(){return _testDataMatrix;}
map<string, int> Dataset::getLabelMap(){return _labelMap;}
map<string, string> Dataset::getTo39PhonemeMap(){return _To39PhonemeMap;}

//Load function
void Dataset::loadTo39PhonemeMap(const char* mapFilePath){
	ifstream fin(mapFilePath);
	if(!fin) cout<<"Can't open the file!\n";
	string s, sKey, sVal;//For map
	while(getline(fin, s)){
		 int pos = 0;
		 int initialPos = 0;
		int judge = 1;
		while(pos!=string::npos){
				
			pos = s.find("\t", initialPos);
			if(judge==1) sKey = s.substr(initialPos, pos-initialPos);
			else
			{
				sVal = s.substr(initialPos, pos-initialPos);
		//		cout<<sKey<<" "<<sVal<<endl;
				_To39PhonemeMap.insert(pair<string,string>(sKey,sVal));
			}
			initialPos = pos+1;
//			pos=s.find("\t", initialPos);
			judge++;
		}
	}
	fin.close();
}

//Print function
void Dataset::printTo39PhonemeMap(map<string, string> Map){
	map<string, string>::iterator MapIter;
	for(MapIter = Map.begin();MapIter!=Map.end();MapIter++){
		cout<<MapIter->first<<"\t"<<MapIter->second<<endl;	
	}
}	
void   Dataset::printLabelMap(map<string, int> Map){
	map<string, int>::iterator labelMapIter;
	for(labelMapIter = Map.begin();labelMapIter!=Map.end();labelMapIter++){
		cout<<labelMapIter->first<<" "<<labelMapIter->second<<endl;
	}
	
}
