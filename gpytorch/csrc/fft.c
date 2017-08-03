#include <TH/TH.h>
#include <fftw3.h>

// Adapted from
// https://github.com/mbhenaff/spectral-lib/blob/master/cuda/cufft.cpp
// http://www.fftw.org/fftw3_doc/Advanced-Complex-DFTs.html
// https://github.com/koraykv/torch-fftw/tree/master/lib/thfftw
int fft1_r2c(THFloatTensor *input, THFloatTensor *output)
{
  // Get n - batch size
  // Get d - number of dimensions
  THArgCheck(THFloatTensor_nDimension(input) == 2,  2, "Input tensor must be 2 dimensional (n x d)");
  THArgCheck(THFloatTensor_isContiguous(input), 2, "Input tensor must be contiguous");
  int n = (int) THFloatTensor_size(input, 0);
  int d = (int) THFloatTensor_size(input, 1);

  THArgCheck(THFloatTensor_nDimension(output) == 3, 2, "Output tensor must be 3 dimensional (n x d x 2)");
  THArgCheck(THFloatTensor_isContiguous(output), 2, "Output tensor must be contiguous");
  THArgCheck(THFloatTensor_size(output, 0) == n, 2, "The first dimension of the output tensor should be n");
  THArgCheck(THFloatTensor_size(output, 1) == (d / 2) + 1, 2, "The second dimension of the output tensor should be (d/2 + 1)");
  THArgCheck(THFloatTensor_size(output, 2) == 2, 2, "The last dimension of the output tensor should be 2");

  // raw pointers
  float *input_data = THFloatTensor_data(input);
  fftwf_complex *output_data = (fftwf_complex*)THFloatTensor_data(output);

  int rank = 1;
  int stride = 1;
  int size[1] = {d};
  int iDist = d;
  int oDist = d / 2 + 1;

  fftwf_plan plan = fftwf_plan_many_dft_r2c(rank, size, n,
      input_data, size, stride, iDist,
      output_data, size, stride, oDist,
      FFTW_ESTIMATE);
  fftwf_execute(plan);

  // clean up
  fftwf_destroy_plan(plan);
  return 0;
}


int fft1_c2r(THFloatTensor *input, THFloatTensor *output)
{
  // Get n - batch size
  // Get d - number of dimensions
  THArgCheck(THFloatTensor_nDimension(output) == 2,  2, "Input tensor must be 2 dimensional (n x d)");
  THArgCheck(THFloatTensor_isContiguous(output), 2, "Input tensor must be contiguous");
  int n = (int) THFloatTensor_size(output, 0);
  int d = (int) THFloatTensor_size(output, 1);

  THArgCheck(THFloatTensor_nDimension(input) == 3, 2, "Output tensor must be 3 dimensional (n x d x 2)");
  THArgCheck(THFloatTensor_isContiguous(input), 2, "Output tensor must be contiguous");
  THArgCheck(THFloatTensor_size(input, 0) == n, 2, "The first dimension of the input tensor should be n");
  THArgCheck(THFloatTensor_size(input, 1) == (d / 2) + 1, 2, "The second dimension of the input tensor should be (d/2 + 1)");
  THArgCheck(THFloatTensor_size(input, 2) == 2, 2, "The last dimension of the input tensor should be 2");

  // raw pointers
  fftwf_complex *input_data = (fftwf_complex*)THFloatTensor_data(input);
  float *output_data = THFloatTensor_data(output);

  int rank = 1;
  int stride = 1;
  int size[1] = {d};
  int iDist = d / 2 + 1;
  int oDist = d;

  fftwf_plan plan = fftwf_plan_many_dft_c2r(rank, size, n,
      input_data, size, stride, iDist,
      output_data, size, stride, oDist,
      FFTW_ESTIMATE);
  fftwf_execute(plan);

  // clean up
  fftwf_destroy_plan(plan);
  return 0;
}
