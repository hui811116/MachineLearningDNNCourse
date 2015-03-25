#include <device_matrix.h>
#include <device_arithmetic.h>
#include <device_math.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

typedef device_matrix<float> mat;

extern mat operator & (const mat& lv, const mat& rv){
	assert(lv.size()==rv.size());
	assert( lv.getRows() == rv.getRows() && rv.getCols() == rv.getCols());
	mat result(lv.getRows(),lv.getCols());
	thrust::device_ptr<float> dv1(lv.getData());
	thrust::device_ptr<float> dv2(rv.getData());
	thrust::transform(dv1,dv1 + lv.getRows() * lv.getCols(), dv2 , result.getData(), thrust::multiplies());
	return result;
}