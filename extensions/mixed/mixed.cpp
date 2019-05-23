#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/DeviceGuard.h>
#include <cublas_v2.h>


torch::Tensor addmm(float alpha, const torch::Tensor& a_, const torch::Tensor& b_, float beta, at::optional<torch::Tensor> c_=at::nullopt){
    AT_CHECK(a_.is_cuda() && b_.is_cuda(), "arg0 and arg1 must be CUDA Tensors");
    AT_CHECK(a_.device() == b_.device(), "arg0 and arg1 must be on the same device");
    torch::Tensor a = (a_.type().scalarType() == torch::ScalarType::Half) ? a_ : a_.to(torch::kFloat16);
    torch::Tensor b = (b_.type().scalarType() == torch::ScalarType::Half) ? b_ : b_.to(torch::kFloat16);
    torch::Tensor c;
    at::OptionalDeviceGuard guard(a_.device());

    int m = b.size(1);
    int k = b.size(0);
    int n = a.size(0);

    AT_CHECK(beta == 0.f || c_.has_value());

    if (c_.has_value()){
        c = *c_;
        AT_CHECK(c.is_cuda() && c.type().scalarType() == torch::ScalarType::Float, "arg2 must be a CUDA Float32 tensor")
        AT_CHECK(c.device() == a.device(), "arg2 must be on the same device as arg0 and arg1")
    }
    else{
        c = torch::empty({n, m}, torch::dtype(torch::kFloat32)
                                       .layout(torch::kStrided)
                                       .device(torch::kCUDA, a.device().index())
                                       .requires_grad(a.requires_grad() || b.requires_grad()));
    }

    auto blasHandle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetMathMode(blasHandle, CUBLAS_TENSOR_OP_MATH);

    // cublas assumes column major so we need to do C' = B'A'
    cublasStatus_t status = cublasGemmEx(blasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                         m, n, k, &alpha,
                                         b.data<at::Half>(), CUDA_R_16F, m,
                                         a.data<at::Half>(), CUDA_R_16F, k,
                                         &beta,
                                         c.data<float>(), CUDA_R_32F, m,
                                         CUDA_R_32F, CUBLAS_GEMM_DFALT_TENSOR_OP);
    AT_CHECK(status == 0, "cublasGemmEx error");
    return c;
}

torch::Tensor baddbmm(float alpha, const torch::Tensor& a_, const torch::Tensor& b_, float beta, at::optional<torch::Tensor> c_=at::nullopt){
    AT_CHECK(a_.is_cuda() && b_.is_cuda(), "arg0 and arg1 must be CUDA Tensors");
    AT_CHECK(a_.device() == b_.device(), "arg0 and arg1 must be on the same device");
    torch::Tensor a = (a_.type().scalarType() == torch::ScalarType::Half) ? a_ : a_.to(torch::kFloat16);
    torch::Tensor b = (b_.type().scalarType() == torch::ScalarType::Half) ? b_ : b_.to(torch::kFloat16);
    torch::Tensor c;
    at::OptionalDeviceGuard guard(a_.device());

    int batch_size = b.size(0);
    int m = b.size(2);
    int k = b.size(1);
    int n = a.size(1);

    AT_CHECK(beta == 0.f || c_.has_value());

    if (c_.has_value()){
        c = *c_;
        AT_CHECK(c.is_cuda() && c.type().scalarType() == torch::ScalarType::Float, "arg2 must be a CUDA Float32 tensor")
        AT_CHECK(c.device() == a.device(), "arg2 must be on the same device as arg0 and arg1")
    }
    else{
        c = torch::empty({batch_size, n, m}, torch::dtype(torch::kFloat32)
                                                  .layout(torch::kStrided)
                                                  .device(torch::kCUDA, a.device().index())
                                                  .requires_grad(a.requires_grad() || b.requires_grad()));
    }

    auto blasHandle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetMathMode(blasHandle, CUBLAS_TENSOR_OP_MATH);

    // cublas assumes column major so we need to do C' = B'A'
    cublasStatus_t status;
    for (int i = 0; i < batch_size; i++){
        status = cublasGemmEx(blasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                              m, n, k, &alpha,
                              b[i].data<at::Half>(), CUDA_R_16F, m,
                              a[i].data<at::Half>(), CUDA_R_16F, k,
                              &beta,
                              c[i].data<float>(), CUDA_R_32F, m,
                              CUDA_R_32F, CUBLAS_GEMM_DFALT_TENSOR_OP);
        AT_CHECK(status == 0, "cublasGemmBatchedEx error");
    }
    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("addmm", &addmm);
  m.def("baddbmm", &baddbmm);
}
