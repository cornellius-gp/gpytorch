#include "gpytorch_solver_cuda.h"


std::tuple<at::Tensor, at::Tensor> batch_symeig_cuda(at::Tensor mat) {
  CHECK_CUDA(mat);
  CHECK_TYPE(mat);
  CHECK_CONTIGUOUS(mat);
  CHECK_SQUARE(mat);

  auto batch_size = batchCount(mat);
  at::Tensor flattened_mat = mat.reshape({batch_size, mat.size(-2), mat.size(-1)});

  at::Tensor evals;
  at::Tensor evecs;
  std::tie(evals, evecs) = _batch_flattened_symeig_cuda(flattened_mat);

  at::IntList evecs_shape = mat.sizes();
  at::IntList evals_shape = at::IntList(evecs_shape.data(), evecs_shape.size() - 1);
  evecs.resize_(evecs_shape);
  evals.resize_(evals_shape);
  return std::make_tuple(evals, evecs);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("batch_symeig_cuda", &batch_symeig_cuda, "Batch symeig solver (CUDA)");
}
