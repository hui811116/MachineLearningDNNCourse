#include <iostream>
#include <string>
#include <vector>
#include <device_matrix.h>
#include "dnn.h"
#include "dataset.h"

using namespace std;

typedef device_matrix<float> mat;

template <typename T>
void randomInit(device_matrix<T>& m) {
	T* h_data = new T [m.size()];
	for (int i=0; i<m.size(); ++i)
		h_data[i] = rand() / (T) RAND_MAX;
	cudaMemcpy(m.getData(), h_data, m.size() * sizeof(T), cudaMemcpyHostToDevice);
	delete [] h_data;
}

int main(int argc, char** argv){
	cout << "Test!\n";
	vector<size_t> dimensions;
	dimensions.push_back(4);
	dimensions.push_back(7);
	dimensions.push_back(5);

	Dataset dataset;
	DNN dnn(dataset, 3, 0.1, dimensions, ALL);
	
	return 0;
}
