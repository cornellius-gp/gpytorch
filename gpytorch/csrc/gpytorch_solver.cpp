#include "gpytorch_solver.h"


std::tuple<at::Tensor, at::Tensor> batch_symeig(at::Tensor mat) {
  CHECK_CONTIGUOUS(mat);
  CHECK_SQUARE(mat);

  auto batch_size = batchCount(mat);
  at::Tensor && evals = at::empty({batch_size, mat.size(-2)}, mat.options());
  at::Tensor && evecs = at::empty({batch_size, mat.size(-2), mat.size(-1)}, mat.options());
  at::Tensor && flattened_mat = mat.reshape({batch_size, mat.size(-2), mat.size(-1)});

  #if defined(_OPENMP)
    #pragma omp parallel for
  #endif
  for (int64_t i = 0; i < batch_size; i++) {
    at::Tensor && sub_evals = evals.select(0, i);
    at::Tensor && sub_evecs = evecs.select(0, i);
    const at::Tensor && sub_mat = flattened_mat.select(0, i);
    at::_th_symeig_out(sub_evals, sub_evecs, sub_mat, true, true);
  }

  at::IntList evecs_shape = mat.sizes();
  at::IntList evals_shape = at::IntList(evecs_shape.data(), evecs_shape.size() - 1);
  evecs.resize_(evecs_shape);
  evals.resize_(evals_shape);
  return std::make_tuple(evals, evecs);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("batch_symeig", &batch_symeig, "Batch symeig solver");
}
