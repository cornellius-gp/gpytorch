#include "_gpytorch_solver_base.h"
#define IS_FLOAT(x) (x.type().scalarType() == at::ScalarType::Float)
#define IS_DOUBLE(x) (x.type().scalarType() == at::ScalarType::Double)
#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_TYPE(x) AT_CHECK((IS_FLOAT(x)) || (IS_DOUBLE(x)), #x " must be a Double or Float tensor")


// Actual function definition
std::tuple<at::Tensor, at::Tensor> batch_symeig_cuda(at::Tensor mat);
std::tuple<at::Tensor, at::Tensor> _batch_flattened_symeig_cuda(at::Tensor flattened_mat);
