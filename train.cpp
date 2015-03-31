#include "parser.h"
#include "dnn.h"
#include "dataset.h"
#include <iostream>
#include <string>
#include <vector>
#include <ctime>
#include <cstdlib>

using namespace std;

void myUsage(){cerr<<"$cmd [inputfile] [testfile] [labelfile] --labeldim [] --phonenum [] --trainnum [] --testnum [] --labelnum [] --inputdim [] --outputdim []"<<endl;}

int main(int argc,char** argv){
	srand(time(0));
	PARSER p;
	p.addMust("trainFilename",false);
	p.addMust("testFilename",false);
	p.addMust("labelFilename",false);
	p.addOption("--labeldim",true);
	p.addOption("--phonenum",true);
	p.addOption("--trainnum",true);
	p.addOption("--testnum",true);
	p.addOption("--labelnum",true);
	p.addOption("--inputdim",true);
	p.addOption("--outputdim",true);
	p.addOption("--rate",true);
	p.addOption("--segment",true);
	p.addOption("--batchsize",true);
	string trainF,testF,labelF;
	size_t labdim,phonenum,trainnum,testnum,labelnum,indim,outdim,b_size=500;
	float rate=0.1,segment=0.8;
	if(!p.read(argc,argv)){
		myUsage();
		return 1;
	}
	p.getString("trainfilename",trainF);
	p.getString("testfilename",testF);
	p.getString("labelFilename",labelF);
	p.getNum("--labeldim",labdim);
	p.getNum("--phonenum",phonenum);
	p.getNum("--trainnum",trainnum);
	p.getNum("--testnum",testnum);
	p.getNum("--labelnum",labelnum);
	p.getNum("--inputdim",indim);
	p.getNum("--outputdim",outdim);
	p.getNum("--rate",rate);
	p.getNum("--segment",segment);
	p.getNum("--batchsize",b_size);
	p.print();
	Dataset dataset = Dataset(trainF.c_str(),trainnum,testF.c_str(),testnum,labelF.c_str(),labelnum,labdim,indim,outdim,phonenum);
	dataset.dataSegment(segment);
	vector<size_t>dim;
	dim.push_back(indim);
	dim.push_back(128);
	dim.push_back(outdim);
	DNN dnn(&dataset,rate,dim,BATCH);
	dnn.train(b_size,100000);
	dnn.save("hui.mdl");
	cout<<"end of training!";
	return 0;
}

