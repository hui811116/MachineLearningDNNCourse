#include <iostream>
#include <string>
#include <vector>
#include "dnn.h"
#include "dataset.h"
#include "parser.h"

using namespace std;

void myUsage(){
    cout<<" USAGE: predict [inputFile] [labelFile] [modelFile] --inputDim --outputDim"<<endl;
}

int main (int argc, char* argv){
    if(argc<4){myUsage();return 0;}
    PARSER cmd;
    cmd.addMust("inputfile",false);
    cmd.addMust("labelfile",false);
    cmd.addMust("testfile",false);
    cmd.addOption("rate",true);
    cmd.addOption("epoch",true);

    cmd.read(argc,argv);

    return 0;
}
