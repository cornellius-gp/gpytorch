#include "_gpytorch_solver_base.h"
#define CHECK_INPUT(x) CHECK_CONTIGUOUS(x); CHECK_SQUARE(x);


// Actual function definitions
std::tuple<at::Tensor, at::Tensor> batch_symeig(at::Tensor mat);
