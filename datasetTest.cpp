#include <iostream>
#include <fstream>
#include "dataset.h"

int main()
{
	size_t  phonemeNum = 39;
	size_t dataNum = 180406;
	const char* inputName = "/home/larry/Documents/data/MLDS_HW1_RELEASE_v1/mfcc/test.ark";	
	Dataset test = Dataset(inputName, dataNum, phonemeNum);

	return 0;
}
