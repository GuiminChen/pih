#include "pih/backend/cuda/gemm_plan.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>

#include <cublasLt.h>

namespace pih {
namespace {

Status cublas_status(cublasStatus_t status, const char* operation) {
  if (status == CUBLAS_STATUS_SUCCESS) {
    return Status::Ok();
  }
  std::string message(operation);
  message.append(" failed with cuBLAS status ");
  message.append(std::to_string(static_cast<int>(status)));
  if (status == CUBLAS_STATUS_ALLOC_FAILED) {
    return Status::ResourceExhausted(std::move(message));
  }
  if (status == CUBLAS_STATUS_INVALID_VALUE) {
    return Status::InvalidArgument(std::move(message));
  }
  if (status == CUBLAS_STATUS_NOT_SUPPORTED ||
      status == CUBLAS_STATUS_ARCH_MISMATCH) {
    return Status::Unavailable(std::move(message));
  }
  return Status::Internal(std::move(message));
}

bool is_matrix(const TensorView& view, std::uint64_t rows, std::uint64_t cols,
               DType dtype) {
  return view.dtype() == dtype && view.rank() == 2 &&
         view.dim(0) == rows && view.dim(1) == cols &&
         view.stride(1) == 1 && view.stride(0) == cols &&
         view.device().type() == DeviceType::kCuda && view.generation() != 0;
}

bool overlaps(const TensorView& lhs, const TensorView& rhs) {
  const auto left = reinterpret_cast<std::uintptr_t>(lhs.data());
  const auto right = reinterpret_cast<std::uintptr_t>(rhs.data());
  const auto maximum = std::numeric_limits<std::uintptr_t>::max();
  if (lhs.byte_span() > maximum - left || rhs.byte_span() > maximum - right) {
    return true;
  }
  const auto left_end = left + static_cast<std::uintptr_t>(lhs.byte_span());
  const auto right_end = right + static_cast<std::uintptr_t>(rhs.byte_span());
  return left < right_end && right < left_end;
}

}  // namespace

struct GemmPlan::Impl final {
  cublasLtHandle_t handle = nullptr;
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatrixLayout_t a_layout = nullptr;
  cublasLtMatrixLayout_t b_layout = nullptr;
  cublasLtMatrixLayout_t c_layout = nullptr;
  cublasLtMatmulAlgo_t algorithm{};

  ~Impl() {
    if (c_layout != nullptr) cublasLtMatrixLayoutDestroy(c_layout);
    if (b_layout != nullptr) cublasLtMatrixLayoutDestroy(b_layout);
    if (a_layout != nullptr) cublasLtMatrixLayoutDestroy(a_layout);
    if (operation != nullptr) cublasLtMatmulDescDestroy(operation);
    if (handle != nullptr) cublasLtDestroy(handle);
  }
};

GemmPlan::GemmPlan(std::uint64_t m, std::uint64_t n, std::uint64_t k,
                   std::uint64_t workspace_bytes, std::int32_t algorithm_id,
                   DType output_dtype,
                   std::unique_ptr<Impl> impl)
    : m_(m), n_(n), k_(k), workspace_bytes_(workspace_bytes),
      algorithm_id_(algorithm_id), output_dtype_(output_dtype),
      impl_(std::move(impl)) {}

GemmPlan::~GemmPlan() = default;

Result<std::unique_ptr<GemmPlan>> GemmPlan::Create(
    std::uint64_t m, std::uint64_t n, std::uint64_t k,
    std::uint64_t max_workspace_bytes, DType output_dtype) {
  constexpr auto kMax = static_cast<std::uint64_t>(
      std::numeric_limits<std::int32_t>::max());
  if (m == 0 || n == 0 || k == 0) {
    return Status::InvalidArgument("BF16 GEMM dimensions must be nonzero");
  }
  if (output_dtype != DType::kBFloat16 && output_dtype != DType::kFloat32) {
    return Status::InvalidArgument("BF16 GEMM output dtype is unsupported");
  }
  if (m > kMax || n > kMax || k > kMax ||
      max_workspace_bytes > std::numeric_limits<std::size_t>::max()) {
    return Status::InvalidArgument("BF16 GEMM plan exceeds bounded dimensions");
  }

  auto impl = std::make_unique<Impl>();
  auto status = cublasLtCreate(&impl->handle);
  if (status != CUBLAS_STATUS_SUCCESS) return cublas_status(status, "cublasLtCreate");
  status = cublasLtMatmulDescCreate(&impl->operation, CUBLAS_COMPUTE_32F,
                                    CUDA_R_32F);
  if (status != CUBLAS_STATUS_SUCCESS) return cublas_status(status, "matmul descriptor");
  status = cublasLtMatrixLayoutCreate(&impl->a_layout, CUDA_R_16BF, m, k, k);
  if (status != CUBLAS_STATUS_SUCCESS) return cublas_status(status, "A layout");
  // Hugging Face/Safetensors linear weights are stored row-major as
  // [output_features, input_features]. Compute A[m,k] * W[n,k]^T.
  status = cublasLtMatrixLayoutCreate(&impl->b_layout, CUDA_R_16BF, n, k, k);
  if (status != CUBLAS_STATUS_SUCCESS) return cublas_status(status, "B layout");
  const auto cuda_output_dtype =
      output_dtype == DType::kFloat32 ? CUDA_R_32F : CUDA_R_16BF;
  status = cublasLtMatrixLayoutCreate(&impl->c_layout, cuda_output_dtype, m, n,
                                      n);
  if (status != CUBLAS_STATUS_SUCCESS) return cublas_status(status, "C layout");

  const cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
  for (auto layout : {impl->a_layout, impl->b_layout, impl->c_layout}) {
    status = cublasLtMatrixLayoutSetAttribute(
        layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
    if (status != CUBLAS_STATUS_SUCCESS) return cublas_status(status, "row-major layout");
  }
  const cublasOperation_t transpose_b = CUBLAS_OP_T;
  status = cublasLtMatmulDescSetAttribute(
      impl->operation, CUBLASLT_MATMUL_DESC_TRANSB, &transpose_b,
      sizeof(transpose_b));
  if (status != CUBLAS_STATUS_SUCCESS) {
    return cublas_status(status, "weight transpose");
  }

  cublasLtMatmulPreference_t preference = nullptr;
  status = cublasLtMatmulPreferenceCreate(&preference);
  if (status != CUBLAS_STATUS_SUCCESS) return cublas_status(status, "preference");
  const std::size_t workspace_limit = static_cast<std::size_t>(max_workspace_bytes);
  status = cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_limit,
      sizeof(workspace_limit));
  if (status != CUBLAS_STATUS_SUCCESS) {
    cublasLtMatmulPreferenceDestroy(preference);
    return cublas_status(status, "workspace preference");
  }
  cublasLtMatmulHeuristicResult_t heuristic{};
  int returned = 0;
  status = cublasLtMatmulAlgoGetHeuristic(
      impl->handle, impl->operation, impl->a_layout, impl->b_layout,
      impl->c_layout, impl->c_layout, preference, 1, &heuristic, &returned);
  cublasLtMatmulPreferenceDestroy(preference);
  if (status != CUBLAS_STATUS_SUCCESS) return cublas_status(status, "algorithm heuristic");
  if (returned != 1 || heuristic.state != CUBLAS_STATUS_SUCCESS) {
    return Status::Unavailable("no compatible frozen BF16 GEMM algorithm");
  }
  impl->algorithm = heuristic.algo;
  std::int32_t algorithm_id = -1;
  std::size_t written = 0;
  status = cublasLtMatmulAlgoConfigGetAttribute(
      &impl->algorithm, CUBLASLT_ALGO_CONFIG_ID, &algorithm_id,
      sizeof(algorithm_id), &written);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return cublas_status(status, "algorithm identity");
  }
  if (written != sizeof(algorithm_id) || algorithm_id < 0) {
    return Status::Internal("cuBLASLt returned an invalid algorithm identity");
  }
  return std::unique_ptr<GemmPlan>(new GemmPlan(
      m, n, k, heuristic.workspaceSize, algorithm_id, output_dtype,
      std::move(impl)));
}

Status GemmPlan::execute(const TensorView& a, const TensorView& b,
                         const TensorView& c, void* workspace,
                         std::uint64_t workspace_bytes,
                         cudaStream_t stream) const {
  if (!is_matrix(a, m_, k_, DType::kBFloat16) ||
      !is_matrix(b, n_, k_, DType::kBFloat16) ||
      !is_matrix(c, m_, n_, output_dtype_) || a.device() != b.device() ||
      a.device() != c.device()) {
    return Status::InvalidArgument("BF16 GEMM tensors do not match frozen plan");
  }
  if (stream == nullptr || stream == cudaStreamLegacy ||
      stream == cudaStreamPerThread) {
    return Status::InvalidArgument("BF16 GEMM requires an explicit engine stream");
  }
  if (overlaps(a, b) || overlaps(a, c) || overlaps(b, c)) {
    return Status::InvalidArgument("BF16 GEMM tensors must not overlap");
  }
  if (workspace_bytes < workspace_bytes_ ||
      (workspace_bytes_ != 0 && workspace == nullptr)) {
    return Status::InvalidArgument("BF16 GEMM workspace is too small");
  }
  const float alpha = 1.0F;
  const float beta = 0.0F;
  const auto status = cublasLtMatmul(
      impl_->handle, impl_->operation, &alpha, a.data(), impl_->a_layout,
      b.data(), impl_->b_layout, &beta, c.data(), impl_->c_layout, c.data(),
      impl_->c_layout, &impl_->algorithm, workspace,
      static_cast<std::size_t>(workspace_bytes_), stream);
  return cublas_status(status, "cublasLtMatmul");
}

}  // namespace pih
