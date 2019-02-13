#include "gpytorch_solver_cuda.h"
#include <cuda_runtime.h>
#include <cusolverDn.h>


// Parameters
const double tol = 1.e-7;
const int max_sweeps = 15;
const int sort_eig = 1;
const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
const cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;


/* Functions for computing the buffer size */
template <typename scalar_t>
cusolverStatus_t _compute_buffer_size(
    cusolverDnHandle_t &cusolver_h, int64_t batch_size, int64_t matrix_size,
    scalar_t *evecs_data, scalar_t *evals_data, int *lwork, syevjInfo_t &syevj_params
);

template <>
cusolverStatus_t _compute_buffer_size<float>(
    cusolverDnHandle_t &cusolver_h, int64_t batch_size, int64_t matrix_size,
    float *evecs_data, float *evals_data, int *lwork, syevjInfo_t &syevj_params
) {
  return cusolverDnSsyevjBatched_bufferSize(
    cusolver_h, jobz, uplo, matrix_size, evecs_data, matrix_size, evals_data, lwork, syevj_params, batch_size
  );
}

template <>
cusolverStatus_t _compute_buffer_size<double>(
    cusolverDnHandle_t &cusolver_h, int64_t batch_size, int64_t matrix_size,
    double *evecs_data, double *evals_data, int *lwork, syevjInfo_t &syevj_params
) {
  return cusolverDnDsyevjBatched_bufferSize(
    cusolver_h, jobz, uplo, matrix_size, evecs_data, matrix_size, evals_data, lwork, syevj_params, batch_size
  );
}


/* Functions for computing eigenvectors and eigenvalues */
template <typename scalar_t>
cusolverStatus_t _syevj_batched_solver(
    cusolverDnHandle_t &cusolver_h, int64_t batch_size, int64_t matrix_size,
    scalar_t *evecs_data, scalar_t *evals_data, scalar_t *work, int lwork, int *info,
		syevjInfo_t &syevj_params
);

template <>
cusolverStatus_t _syevj_batched_solver(
    cusolverDnHandle_t &cusolver_h, int64_t batch_size, int64_t matrix_size,
    float *evecs_data, float *evals_data, float *work, int lwork, int *info,
		syevjInfo_t &syevj_params
) {
	return cusolverDnSsyevjBatched(
		cusolver_h, jobz, uplo, matrix_size, evecs_data, matrix_size, evals_data, work, lwork, 
		info, syevj_params, batch_size
	);
}

template <>
cusolverStatus_t _syevj_batched_solver<double>(
    cusolverDnHandle_t &cusolver_h, int64_t batch_size, int64_t matrix_size,
    double *evecs_data, double *evals_data, double *work, int lwork, int *info,
		syevjInfo_t &syevj_params
) {
	return cusolverDnDsyevjBatched(
		cusolver_h, jobz, uplo, matrix_size, evecs_data, matrix_size, evals_data, work, lwork, 
		info, syevj_params, batch_size
	);
}


/* Helper function - takes template scalar type */
template <typename scalar_t>
void _batch_flattened_symeig_cuda_helper(
    scalar_t *evals_data, scalar_t *evecs_data, int *info, int64_t batch_size, int64_t matrix_size
) {
  // Variables for storage
	cusolverDnHandle_t cusolver_h = NULL;
	cudaStream_t stream = NULL;
	syevjInfo_t syevj_params = NULL;
	cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
	cudaError_t cuda_stat = cudaSuccess;

	// Create cusolver handle, bind a stream
	status = cusolverDnCreate(&cusolver_h);
	assert(CUSOLVER_STATUS_SUCCESS == status);
	cuda_stat = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
	assert(cudaSuccess == cuda_stat);
	status = cusolverDnSetStream(cusolver_h, stream);
	assert(CUSOLVER_STATUS_SUCCESS == status);

	// Configuration of syevj
	status = cusolverDnCreateSyevjInfo(&syevj_params);
	assert(CUSOLVER_STATUS_SUCCESS == status);
	status = cusolverDnXsyevjSetTolerance(syevj_params, tol);
	assert(CUSOLVER_STATUS_SUCCESS == status);
	status = cusolverDnXsyevjSetMaxSweeps(syevj_params, max_sweeps);
	assert(CUSOLVER_STATUS_SUCCESS == status);
	status = cusolverDnXsyevjSetSortEig(syevj_params, sort_eig);
	assert(CUSOLVER_STATUS_SUCCESS == status);

  // Determine optimal size of work array
  int lwork = 0;
  status = _compute_buffer_size<scalar_t>(
    cusolver_h, batch_size, matrix_size, evecs_data, evals_data, &lwork, syevj_params
  );
  assert(CUSOLVER_STATUS_SUCCESS == status);

  // Allocate work array
  scalar_t *work = NULL;
  cuda_stat = cudaMalloc((void**) &work, sizeof(scalar_t)* lwork);
  assert(cuda_stat == cudaSuccess);

  // Compute eigenvectors and eigenvalues
  status = _syevj_batched_solver(
    cusolver_h, batch_size, matrix_size, evecs_data, evals_data, work, lwork, info, syevj_params
  );
  cuda_stat = cudaDeviceSynchronize();
  assert(CUSOLVER_STATUS_SUCCESS == status);
  assert(cudaSuccess == cuda_stat);

  // Cleanup
  if (work) cudaFree(work);
  if (cusolver_h) cusolverDnDestroy(cusolver_h);
  if (stream) cudaStreamDestroy(stream);
  if (syevj_params) cusolverDnDestroySyevjInfo(syevj_params);
}


std::tuple<at::Tensor, at::Tensor> _batch_flattened_symeig_cuda(at::Tensor flattened_mat) {
  // Get constants, create return tensors
  int64_t batch_size = flattened_mat.size(0);
  int64_t matrix_size = flattened_mat.size(1);
  at::Tensor flattened_evals = at::zeros({batch_size, flattened_mat.size(-2)}, flattened_mat.options());
  at::Tensor flattened_evecs = at::clone(flattened_mat);
  at::Tensor info = at::empty({batch_size}, flattened_mat.options().dtype(at::kInt));

  // Perform eigensolve
  switch (flattened_mat.type().scalarType()) {
    case at::ScalarType::Double:
      _batch_flattened_symeig_cuda_helper<double>(
        flattened_evals.data<double>(),
        flattened_evecs.data<double>(),
        info.data<int>(), batch_size, matrix_size
      );
      break;

    case at::ScalarType::Float:
      _batch_flattened_symeig_cuda_helper<float>(
        flattened_evals.data<float>(),
        flattened_evecs.data<float>(),
        info.data<int>(), batch_size, matrix_size
      );
      break;

    default:
      AT_ERROR("This function doesn't handle types other than float and double");
  }

  // Check error status
  if (at::is_nonzero(info)) {
    AT_ERROR("CUSolver did not converge");
  }

  return std::make_tuple(flattened_evals, flattened_evecs);
}
