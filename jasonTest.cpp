#include <iostream>
#include "dataset.h"
using namespace std;
int main(){
	cout << "This is the data set test\n";	
	size_t  phonemeNum = 39;
	size_t dataNum = 50;
	const char* inputName = "/home/larry/Documents/data/MLDS_HW1_RELEASE_v1/mfcc/test.ark";	
	Dataset test = Dataset(inputName, dataNum, phonemeNum);
	cout << "construct dataset done\n";
	
}
