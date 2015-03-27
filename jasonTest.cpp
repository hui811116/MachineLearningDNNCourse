#include <iostream>
#include "dataset.h"
using namespace std;
typedef device_matrix<float> mat;
int main(){
	cout << "This is the data set test\n";	
	size_t  phonemeNum = 39;
	size_t dataNum = 50;
	const char* inputName = "/home/larry/Documents/data/MLDS_HW1_RELEASE_v1/mfcc/test.ark";	
	Dataset test = Dataset(inputName, dataNum, phonemeNum);
	test.dataSegment(0.8);
	cout << "construct dataset done\n";
    // test
	mat batch();
	mat batchLabel();
	test.getBatch(5, batch, batchLabel);
}
