#include <THC/THC.h>
#include <cufft.h>

// this symbol will be resolved automatically from PyTorch libs
extern THCState *state;

// Adapted from
// https://github.com/mbhenaff/spectral-lib/blob/master/cuda/cufft.cpp
int fft1_r2c_cuda(THCudaTensor *input, THCudaTensor *output)
{
  // Get n - batch size
  // Get d - number of dimensions
  THArgCheck(THCudaTensor_nDimension(state, input) == 2,  2, "Input tensor must be 2 dimensional (n x d)");
  THArgCheck(THCudaTensor_isContiguous(state, input), 2, "Input tensor must be contiguous");
  int n = (int) THCudaTensor_size(state, input, 0);
  int d = (int) THCudaTensor_size(state, input, 1);

  THArgCheck(THCudaTensor_nDimension(state, output) == 3, 2, "Output tensor must be 3 dimensional (n x d x 2)");
  THArgCheck(THCudaTensor_isContiguous(state, output), 2, "Output tensor must be contiguous");
  THArgCheck(THCudaTensor_size(state, output, 0) == n, 2, "The first dimension of the output tensor should be n");
  THArgCheck(THCudaTensor_size(state, output, 1) == (d / 2) + 1, 2, "The second dimension of the output tensor should be (d/2 + 1)");
  THArgCheck(THCudaTensor_size(state, output, 2) == 2, 2, "The last dimension of the output tensor should be 2");

  // raw pointers
  float *input_data = THCudaTensor_data(NULL, input);
  cuComplex *output_data = (cuComplex*) THCudaTensor_data(NULL, output);

  // execute FFT
  cufftHandle plan;
  cufftPlan1d(&plan, d, CUFFT_R2C, n);
  cufftExecR2C(plan, (cufftReal*) input_data, (cufftComplex*) output_data);

  // clean up
  cufftDestroy(plan);
  return 0;
}


int fft1_c2r_cuda(THCudaTensor *input, THCudaTensor *output)
{
  // Get n - batch size
  // Get d - number of dimensions
  THArgCheck(THCudaTensor_nDimension(state, output) == 2,  2, "Input tensor must be 2 dimensional (n x d)");
  THArgCheck(THCudaTensor_isContiguous(state, output), 2, "Input tensor must be contiguous");
  int n = (int) THCudaTensor_size(state, output, 0);
  int d = (int) THCudaTensor_size(state, output, 1);

  THArgCheck(THCudaTensor_nDimension(state, input) == 3, 2, "Output tensor must be 3 dimensional (n x d x 2)");
  THArgCheck(THCudaTensor_isContiguous(state, input), 2, "Output tensor must be contiguous");
  THArgCheck(THCudaTensor_size(state, input, 0) == n, 2, "The first dimension of the input tensor should be n");
  THArgCheck(THCudaTensor_size(state, input, 1) == (d / 2) + 1, 2, "The second dimension of the input tensor should be (d/2 + 1)");
  THArgCheck(THCudaTensor_size(state, input, 2) == 2, 2, "The last dimension of the input tensor should be 2");

  // raw pointers
  cuComplex *input_data = (cuComplex*) THCudaTensor_data(NULL, input);
  float *output_data = THCudaTensor_data(NULL, output);

  // execute FFT
  cufftHandle plan;
  cufftPlan1d(&plan, d, CUFFT_C2R, n);
  cufftExecC2R(plan, (cufftComplex*) input_data, (cufftReal*) output_data);

  // clean up
  cufftDestroy(plan);
  return 0;
}
