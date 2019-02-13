#include <torch/extension.h>
#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_SQUARE(x) AT_CHECK(x.size(-2) == x.size(-1), #x " must be square")


// Taken from https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/LinearAlgebraUtils.h
static inline int64_t batchCount(const at::Tensor& batched_matrices) {
  int64_t result = 1;
  for (int64_t i = 0; i < batched_matrices.ndimension() - 2; i++) {
    result *= batched_matrices.size(i);
  }
  return result;
}
