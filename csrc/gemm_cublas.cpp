/*
 * Copyright (c) 2025, Tri Dao.
 */
#include <torch/python.h>
// #include <Python.h>  // for PyModuleDef_HEAD_INIT
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>  // for at::cuda::getCurrentDeviceProperties and at::cuda::getCurrentCUDABlasLtHandle
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/Exceptions.h>  // For TORCH_CUDABLAS_CHECK

// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <cublas_v2.h>
// #include <cublasLt.h>

#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ "); (got ", x.sizes(), ")")

// Copied from https://pytorch.org/tutorials/advanced/cpp_custom_ops.html
extern "C" {
  /* Creates a dummy empty _C module that can be imported from Python.
     The import from Python will load the .so consisting of this file
     in this extension, so that the TORCH_LIBRARY static initializers
     below are run. */
  PyObject* PyInit__C(void)
  {
    static struct PyModuleDef module_def = {
      PyModuleDef_HEAD_INIT,
      "_C",   /* name of module */
      NULL,   /* module documentation, may be NULL */
      -1,     /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
      NULL,   /* methods */
    };
    return PyModule_Create(&module_def);
  }
}

uint32_t _getAlignment(uintptr_t address) {
  // alignment are in bytes
  for (uint32_t alignment = 256; ; alignment /= 2) {
    if (address % alignment == 0) { return alignment; }
  }
}

at::Tensor
gemm_impl(const at::Tensor& A_, const at::Tensor& B_,
          std::optional<at::Tensor> C_=std::nullopt,
          std::optional<at::Tensor> out_=std::nullopt,
          std::optional<at::ScalarType> out_dtype_=std::nullopt,
          int64_t heuristic_=-1) {
  int heuristic = heuristic_;
  bool C_rowmajor = out_.has_value()
    ? out_.value().stride(1) == 1
    : (C_.has_value() ? C_.value().stride(1) == 1 : true);
  // If C_rowmajor, instead of compute out = A @ B, we compute out.T = B.T @ A.T
  at::Tensor A = !C_rowmajor ? A_ : B_.transpose(0, 1);
  at::Tensor B = !C_rowmajor ? B_ : A_.transpose(0, 1);
  const int64_t m = A.size(0);
  const int64_t k = A.size(1);
  const int64_t n = B.size(1);

  auto input_type = A.scalar_type();
  TORCH_CHECK(input_type == at::ScalarType::BFloat16 || input_type == at::ScalarType::Half || input_type == at::ScalarType::Float, "Input must be of type BF16, FP16, or FP32");
  TORCH_CHECK(B.scalar_type() == input_type);
  TORCH_CHECK(A.is_cuda());
  TORCH_CHECK(B.is_cuda());
  CHECK_SHAPE(A, m, k);
  CHECK_SHAPE(B, k, n);
  TORCH_CHECK(A.stride(0) == 1 || A.stride(1) == 1, "Input must be contiguous in dim 0 or dim 1");
  TORCH_CHECK(B.stride(0) == 1 || B.stride(1) == 1, "Input must be contiguous in dim 0 or dim 1");
  bool A_rowmajor = A.stride(1) == 1;
  bool B_rowmajor = B.stride(1) == 1;

  at::Tensor C;
  if (C_.has_value()) {
    C = C_.value();
    TORCH_CHECK(C.scalar_type() == input_type || C.scalar_type() == at::ScalarType::Float, "C must be of the same type as A and B, or of type FP32");
    TORCH_CHECK(C.is_cuda());
    if (C_rowmajor) { C = C.transpose(0, 1); }  // Now C is colmajor
    CHECK_SHAPE(C, m, n);
    TORCH_CHECK(C.stride(0) == 1);
  }

  at::Tensor out;
  auto opts = A.options();
  if (!out_.has_value()) {
    auto out_type = C_.has_value() ? C_.value().scalar_type() : out_dtype_.value_or(input_type);
    out = at::empty({n, m}, opts.dtype(out_type)).transpose(0, 1);  // Colmajor
  } else {
    out = out_.value();
    TORCH_CHECK(out.scalar_type() == input_type || out.scalar_type() == at::ScalarType::Float, "out must be of the same type as A and B, or of type FP32");
    TORCH_CHECK(out.is_cuda());
    if (C_rowmajor) { out = out.transpose(0, 1); }  // Now out is colmajor
    CHECK_SHAPE(out, m, n);
    TORCH_CHECK(out.stride(0) == 1);
  }
  auto out_type = out.scalar_type();

  // Otherwise the kernel will be launched from cuda:0 device
  at::cuda::CUDAGuard device_guard{A.device()};

  size_t workspaceSize = 1024 * 1024 * (at::cuda::getCurrentDeviceProperties()->major >= 9 ? 32 : 4);
  auto lt_workspace = at::empty({static_cast<int64_t>(workspaceSize)}, opts.dtype(torch::kUInt8));

  const void* A_ptr = A.data_ptr();
  const void* B_ptr = B.data_ptr();
  const void* C_ptr = C_.has_value() ? C.data_ptr() : nullptr;
  void* out_ptr = out.data_ptr();

  float alpha = 1.f;
  float beta = C_.has_value() ? 1.f : 0.f;
  cudaDataType_t abType = input_type == at::ScalarType::BFloat16 ? CUDA_R_16BF : (input_type == at::ScalarType::Half ? CUDA_R_16F : CUDA_R_32F);
  cudaDataType_t cType = out_type == at::ScalarType::BFloat16 ? CUDA_R_16BF : (out_type == at::ScalarType::Half ? CUDA_R_16F : CUDA_R_32F);
  cublasOperation_t transa = A_rowmajor ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transb = B_rowmajor ? CUBLAS_OP_T : CUBLAS_OP_N;

  cublasLtMatmulDescOpaque_t operationDesc = {};
  cublasLtMatrixLayoutOpaque_t Adesc = {}, Bdesc = {}, Cdesc = {}, Ddesc = {};
  cublasLtMatmulPreferenceOpaque_t preference = {};

  // Create operation descriptor; see cublasLtMatmulDescAttributes_t for details about defaults.
  cublasLtHandle_t ltHandle = at::cuda::getCurrentCUDABlasLtHandle();
  auto compute_type = input_type == at::ScalarType::Float && at::globalContext().allowTF32CuBLAS() ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
  TORCH_CUDABLAS_CHECK(cublasLtMatmulDescInit(&operationDesc, compute_type, CUDA_R_32F /*scale_type*/));
  TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
  TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(&operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transa)));

  TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutInit(&Adesc, abType, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, A_rowmajor ? A.stride(0) : A.stride(1) /*lda*/));
  TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutInit(&Bdesc, abType, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, B_rowmajor ? B.stride(0) : B.stride(1) /*ldb*/));
  TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutInit(&Ddesc, cType, m, n, out.stride(1) /*ldd*/));
  TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutInit(&Cdesc, cType, m, n, C_.has_value() ? C.stride(1) : out.stride(1) /*ldc*/));

  TORCH_CUDABLAS_CHECK(cublasLtMatmulPreferenceInit(&preference));
  TORCH_CUDABLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(&preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

  uint32_t a_alignment = _getAlignment(reinterpret_cast<uintptr_t>(A_ptr));
  uint32_t b_alignment = _getAlignment(reinterpret_cast<uintptr_t>(B_ptr));
  uint32_t c_alignment = _getAlignment(reinterpret_cast<uintptr_t>(C_ptr));
  uint32_t d_alignment = _getAlignment(reinterpret_cast<uintptr_t>(out_ptr));
  TORCH_CUDABLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(&preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, &a_alignment, sizeof(a_alignment)));
  TORCH_CUDABLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(&preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, &b_alignment, sizeof(b_alignment)));
  TORCH_CUDABLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(&preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, &c_alignment, sizeof(c_alignment)));
  TORCH_CUDABLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(&preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, &d_alignment, sizeof(d_alignment)));

  constexpr int requestedAlgoCount = 8;
  cublasLtMatmulHeuristicResult_t heuristicResult[requestedAlgoCount] = { 0 };
  if (heuristic >= 0) {  // If heuristic == -1 we don't query
    int returnedResult = 0;
    TORCH_CUDABLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
      ltHandle, &operationDesc, &Adesc, &Bdesc, &Cdesc, &Ddesc, &preference,
      requestedAlgoCount, heuristicResult, &returnedResult));
    if (returnedResult == 0) { TORCH_CUDABLAS_CHECK(CUBLAS_STATUS_NOT_SUPPORTED); }
    heuristic = std::min(heuristic, returnedResult - 1);
  }
  TORCH_CUDABLAS_CHECK(cublasLtMatmul(
    ltHandle,
    &operationDesc,
    &alpha,
    A_ptr,
    &Adesc,
    B_ptr,
    &Bdesc,
    &beta,
    C_ptr,
    &Cdesc,
    out_ptr,
    &Ddesc,
    heuristic >= 0 ? &heuristicResult[heuristic].algo : nullptr,
    (void*) (lt_workspace.data_ptr()),
    workspaceSize,
    c10::cuda::getCurrentCUDAStream()));
  // torch.library doesn't like returning a tensor that aliases an input tensor, so we return None
  // if out was provided.
  return !out_.has_value() ? (C_rowmajor ? out.transpose(0, 1) : out) : at::Tensor{};
}

TORCH_LIBRARY(gemm_cublas, m) {
  m.def("gemm_impl(Tensor A, Tensor B, Tensor? C=None, Tensor(a!)? out=None, ScalarType? out_dtype=None, int heuristic=-1) -> Tensor");
}

// Registers CUDA implementations
TORCH_LIBRARY_IMPL(gemm_cublas, CUDA, m) {
  m.impl("gemm_impl", &gemm_impl);
}
