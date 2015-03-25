#include <iostream>
#include <fstream>
#include <dataset.h>

int main()
{
	const int phonemeNum = 39;
	const int dataNum = 180406;
	const char* inputName = "~/../larry/Documents/data/MLDS_HW1_RELEASE_v1/mfcc/test.ark";	
	Dataset test = Dataset(inputName);

	return 0;
}
