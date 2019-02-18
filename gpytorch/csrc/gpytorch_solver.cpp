#include "gpytorch_solver.h"


std::tuple<at::Tensor, at::Tensor, at::Tensor> batch_svd(at::Tensor mat) {
  auto batch_size = batchCount(mat);
  auto m = mat.size(-2);
  auto n = mat.size(-1);
  auto min_size = (m < n) ? m : n;
  at::Tensor && umats = at::empty({batch_size, m, min_size}, mat.options());
  at::Tensor && svecs = at::empty({batch_size, min_size}, mat.options());
  at::Tensor && vmats = at::empty({batch_size, n, min_size}, mat.options());
  at::Tensor && flattened_mats = mat.reshape({batch_size, mat.size(-2), mat.size(-1)});

  #if defined(_OPENMP)
    #pragma omp parallel for
  #endif
  for (int64_t i = 0; i < batch_size; i++) {
    at::Tensor && umat = umats.select(0, i);
    at::Tensor && svec = svecs.select(0, i);
    at::Tensor && vmat = vmats.select(0, i);
    const at::Tensor && mat = flattened_mats.select(0, i);
    at::_th_svd_out(umat, svec, vmat, mat, true, true);
  }
  umats = umats.transpose(-1, -2);

  return std::make_tuple(umats, svecs, vmats);
}


std::tuple<at::Tensor, at::Tensor> batch_symeig(at::Tensor mat) {
  CHECK_SQUARE(mat);

  auto batch_size = batchCount(mat);
  at::Tensor && evals = at::empty({batch_size, mat.size(-2)}, mat.options());
  at::Tensor && evecs = at::empty({batch_size, mat.size(-2), mat.size(-1)}, mat.options());
  at::Tensor && flattened_mats = mat.reshape({batch_size, mat.size(-2), mat.size(-1)});

  #if defined(_OPENMP)
    #pragma omp parallel for
  #endif
  for (int64_t i = 0; i < batch_size; i++) {
    at::Tensor && sub_evals = evals.select(0, i);
    at::Tensor && sub_evecs = evecs.select(0, i);
    const at::Tensor && mat = flattened_mats.select(0, i);
    at::_th_symeig_out(sub_evals, sub_evecs, mat, true, true);
  }

  at::IntList evecs_shape = mat.sizes();
  at::IntList evals_shape = at::IntList(evecs_shape.data(), evecs_shape.size() - 1);
  evecs.resize_(evecs_shape);
  evals.resize_(evals_shape);
  return std::make_tuple(evals, evecs);
}


/* Binding */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("batch_svd", &batch_svd, "Batch svd");
  m.def("batch_symeig", &batch_symeig, "Batch symeig solver");
}
