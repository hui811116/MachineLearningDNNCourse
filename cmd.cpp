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
	string trainF,testF,labelF;
	size_t labdim,phonenum,trainnum,testnum,labelnum,indim,outdim;
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
	p.print();
	Dataset dataset = Dataset(trainF.c_str(),trainnum,testF.c_str(),testnum,labelF.c_str(),labelnum,labdim,indim,outdim,phonenum);
	dataset.dataSegment(0.8);
	return 0;
}

