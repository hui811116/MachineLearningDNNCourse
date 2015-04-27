#ifndef UTIL_H
#define UTIL_H

#include "mynngen.h"
#include "device_matrix.h"

using namespace std;

typedef device_matrix<float> mat;
extern myNnGen gn;

extern void rand_init(mat& w,float range=1);
extern void rand_norm(mat& w,myNnGen& ran=gn);

extern void pushOne(mat& in);
extern void getBias(mat& out,const mat& w);
extern void replaceBias(mat& w,const mat& bias);
extern void parseDim(string str, vector<size_t>& dim);

#endif
