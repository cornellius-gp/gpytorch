#include <torch/extension.h>

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <cuda_runtime.h>
#include <cusparse.h>


void print_device_memory(float *d_ptr, int len) {
    float *h_ptr = (float *)malloc(len * sizeof(float));

    cudaError_t status = cudaMemcpy(h_ptr, d_ptr, len * sizeof(float), cudaMemcpyDeviceToHost);
    assert(status == cudaSuccess);

    for (int i = 0; i < len; ++i) {
        printf("%f, ", h_ptr[i]);
    }
    printf("\n");

    free(h_ptr);
}

void print_device_memory(long *d_ptr, int len) {
    long *h_ptr = (long *)malloc(len * sizeof(long));
    assert(cudaMemcpy(h_ptr, d_ptr, len * sizeof(long), cudaMemcpyDeviceToHost) == cudaSuccess);

    for (int i = 0; i < len; ++i) {
        printf("%ld, ", h_ptr[i]);
    }
    printf("\n");

    free(h_ptr);
}

/*
CUSA_R_32F

1. Only works for float tensor. No double tensor.
2. All tensors have to be on cuda:0.
3. The matrix has to be lower triangular. No upper triangular matrix.
*/
torch::Tensor sparse_triangular_solve(torch::Tensor mat, torch::Tensor rhs) {
    auto n = mat.size(0);
    auto m = rhs.size(1);
    auto nnz = mat._values().size(0);

    // printf("n = %ld, m = %ld, nnz = %ld\n", n, m, nnz);

    // printf("address of indices %p\n", (void *)mat._indices().data_ptr<long>());
    // print_device_memory(mat._indices().data_ptr<long>(), nnz);
    // print_device_memory(mat._indices().data_ptr<long>() + nnz, nnz);

    // printf("address of values %p\n", (void *)mat._values().data_ptr<float>());
    // print_device_memory(mat._values().data_ptr<float>(), nnz);

    // printf("address of rhs %p\n", (void *)rhs.data_ptr<float>());
    // print_device_memory(rhs.data_ptr<float>(), n * m);

    const float alpha = 1.0;

    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t descr_mat;
    cusparseDnMatDescr_t descr_rhs, descr_ret;

    void *dBuffer = NULL;
    size_t bufferSize = 0;

    cusparseSpSMDescr_t spsmDescr;

    // CHECK_CUSPARSE(cusparseCreate(&handle))
    cusparseCreate(&handle);

    /* descriptor of A */
    // Create sparse matrix A in COO format
    assert(cusparseCreateCoo(
        &descr_mat, n, n, nnz,
        mat._indices().data_ptr<long>(), mat._indices().data_ptr<long>() + nnz, mat._values().data_ptr<float>(),
        CUSPARSE_INDEX_64I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_32F) == CUSPARSE_STATUS_SUCCESS
    );

    // Specify Lower | Upper fill mode.
    cusparseFillMode_t fillmode = CUSPARSE_FILL_MODE_LOWER;
    cusparseSpMatSetAttribute(
        descr_mat, CUSPARSE_SPMAT_FILL_MODE,
        &fillmode, sizeof(fillmode)
    );

    // Specify Unit|Non-Unit diagonal type.
    cusparseDiagType_t diagtype = CUSPARSE_DIAG_TYPE_NON_UNIT;
    cusparseSpMatSetAttribute(
        descr_mat, CUSPARSE_SPMAT_DIAG_TYPE,
        &diagtype, sizeof(diagtype)
    );

    /* descriptor of rhs */
    cusparseCreateDnMat(
        &descr_rhs, n, m, m, rhs.data_ptr<float>(),
        CUDA_R_32F, CUSPARSE_ORDER_ROW
    );

    /* descriptor of ret */
    float *dret = NULL;
    cudaMalloc((void**) &dret, n * m * sizeof(float));

    cusparseCreateDnMat(
        &descr_ret, n, m, m, dret,
        CUDA_R_32F, CUSPARSE_ORDER_ROW
    );

    // Create opaque data structure, that holds analysis data between calls.
    cusparseSpSM_createDescr(&spsmDescr);
 
    // allocate an external buffer for analysis
    cusparseSpSM_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, descr_mat, descr_rhs, descr_ret,
        CUDA_R_32F,
        CUSPARSE_SPSM_ALG_DEFAULT,
        spsmDescr,
        &bufferSize
    );

    // analysis
    cudaMalloc(&dBuffer, bufferSize);
    cusparseSpSM_analysis(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, descr_mat, descr_rhs, descr_ret,
        CUDA_R_32F,
        CUSPARSE_SPSM_ALG_DEFAULT,
        spsmDescr,
        dBuffer
    );

    // execute SpSM
    assert(cusparseSpSM_solve(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, descr_mat, descr_rhs, descr_ret,
        CUDA_R_32F,
        CUSPARSE_SPSM_ALG_DEFAULT,
        spsmDescr) == CUSPARSE_STATUS_SUCCESS
    );

    // printf("address of dret %p\n", (void *)dret);
    // print_device_memory(dret, n * m);

    // destroy matrix/vector descriptors
    cusparseSpSM_destroyDescr(spsmDescr);
    cusparseDestroy(handle);

    // device memory deallocation
    cudaFree(dBuffer);

    /* wrap descr_ret into a dense tensor and return it */
    return torch::from_blob(
        dret,
        {n, m},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0)
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_triangular_solve", &sparse_triangular_solve, "Sparse Triangular Solve");
}
