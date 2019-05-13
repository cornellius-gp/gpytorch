#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>

/**
    Computes matrix multiplication C = A \times B where A and B are fp16 and C is accumulated and returned in fp32
*/
torch::Tensor matmul(const torch::Tensor& a, const torch::Tensor& b){
    AT_CHECK(a.is_cuda() && a.type().scalarType() ==  torch::ScalarType::Half, "arg0 must be a CUDA Half tensor");  
    AT_CHECK(b.is_cuda() && b.type().scalarType() ==  torch::ScalarType::Half, "arg1 must be a CUDA Half tensor");  
    AT_CHECK(a.device() == b.device(), "arg0 and arg1 must be on the same device");  

    auto blasHandle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetMathMode(blasHandle, CUBLAS_TENSOR_OP_MATH);
    int m = b.size(1);
    int k = b.size(0);
    int n = a.size(0);
    torch::Tensor c = torch::empty({m, n}, torch::dtype(torch::kFloat32)
                                                 .layout(torch::kStrided)
                                                 .device(torch::kCUDA, a.device().index())
                                                 .requires_grad(a.requires_grad() || b.requires_grad()));

    // cublasGemmEX computes C = \alpha CUBLAS_OP(A) CUBLAS_OP(B) + \beta C
    float alpha = 1.f;
    float beta = 0.f;
    // cublas assumes column major so we need to do C' = B'A' and tranpose the
    // result
    cublasStatus_t status = cublasGemmEx(blasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                         m, n, k, &alpha,
                                         b.data<at::Half>(), CUDA_R_16F, m,
                                         a.data<at::Half>(), CUDA_R_16F, k,
                                         &beta,
                                         c.data<float>(), CUDA_R_32F, m,
                                         CUDA_R_32F, CUBLAS_GEMM_DFALT_TENSOR_OP);
    AT_CHECK(status == 0, "cublasGemmEx error");
    return c.transpose(0, 1);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("matmul", &matmul);
}
