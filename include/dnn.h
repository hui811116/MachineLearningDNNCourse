#ifndef DNN_H
#define DNN_H
#include <vector>
#include "sigmoid.h"
//#include "dataset.h"
#include <fstream>

using namespace std;
enum method{	ALL,BATCH,ONE		};

class dnn{
public:
	dnn();
	dnn(dataset);
	dnn(ifstream*);
	~dnn();

	void train(dataset,method);
	void predict(dataset,vector<float>&);
	
	void save(ofstream*);

private:
	bool feedForward(dataset,vector<float>&);
	bool backPropagate(dataset);
	vector<sigmoid>* _layer;
	
};


#endif
