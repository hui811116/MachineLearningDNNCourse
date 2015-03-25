#ifndef __DEVICE_BLAS_H_
#define __DEVICE_BLAS_H_

#include <device_matrix.h>

#include <device_vector_operators.h>

#define dvec thrust::device_vector
#define dmat device_matrix

// =====================================
// ===== Matrix - Vector Operators =====
// =====================================

template <typename T>
dmat<T> operator * (const dvec<T>& col_vector, const dvec<T>& row_vector) {
  size_t m = col_vector.size();
  size_t n = row_vector.size();
  dmat<T> result(m, n, 0);
  size_t k = 1;

  // Treat device_vector as an 1 by N matrix
  const T* cv = thrust::raw_pointer_cast(col_vector.data());
  const T* rv = thrust::raw_pointer_cast(row_vector.data());

  T alpha = 1.0, beta = 0.0;

  int lda = m;
  int ldb = 1;
  int ldc = m;

  device_matrix<T>::cublas_gemm(CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, cv, lda, rv, ldb, beta, result.getData(), ldc);

  return result;
}

template <typename T>
dvec<T> operator & (const dvec<T>& x, const dvec<T>& y) {
  assert(x.size() == y.size());
  dvec<T> z(x.size());
  thrust::transform(x.begin(), x.end(), y.begin(), z.begin(), thrust::multiplies<T>());
  return z;
}
/*
// modified
template <typename T>
dmat<T> operator & (const dmat<T>& x, const dmat<T>& y){
  assert(x.size() == y.size());
  assert( x.getRows() == y.getRows() && x.getCols() == y.getCols());
  dmat<T> result(x.getRows(),y.getCols());
  thrust::device_ptr<T> dv1(x.getData());
  thrust::device_ptr<T> dv2(y.getData());
  thrust::transform(dv1,dv1 + x.getRows() * x.getCols(), dv2 , result.getData(), thrust::multiplies<T>());
  return result;
  
}
// until here
*/
template <typename T>
dmat<T> operator * (const dvec<T>& v, const dmat<T>& A) {
  assert(v.size() == A.getRows());
  device_matrix<T> m(1, A.getCols(), 0);

  // u = v*A = trans( trans(A) * trans(v) )
  // And there's nothing to do when a vector is transposed
  
  // cublasOperation_t op = A.isTransposed() ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t op = CUBLAS_OP_T;

  T alpha = 1.0, beta = 0.0;
  device_matrix<T>::cublas_gemv(op, A.getRows(), A.getCols(), alpha, A.getData(), A.getRows(), thrust::raw_pointer_cast(v.data()), 1, beta, m.getData(), 1);

  return m;
}

template <typename T>
dmat<T> operator * (const dmat<T>& A, const dvec<T>& v) {
  assert(A.getCols() == v.size());

  device_matrix<T> m(A.getRows(), 1, 0);

  // cublasOperation_t op = A.isTransposed() ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t op = CUBLAS_OP_N;

  T alpha = 1.0, beta = 0.0;
  device_matrix<T>::cublas_gemv(op, A.getRows(), A.getCols(), alpha, A.getData(), A.getRows(), thrust::raw_pointer_cast(v.data()), 1, beta, m.getData(), 1);

  return m;
}

#undef dvec
#undef dmat

#endif // __DEVICE_BLAS_H_
