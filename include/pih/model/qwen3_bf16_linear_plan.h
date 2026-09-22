#pragma once

#include <array>
#include <cstdint>
#include <memory>

#include <cuda_runtime_api.h>

#include "pih/backend/cuda/gemm_plan.h"
#include "pih/core/result.h"
#include "pih/core/tensor_view.h"
#include "pih/model/qwen3_bf16_linear_shape.h"

namespace pih {

class QwenBf16LinearPlanSet final {
 public:
  static constexpr std::size_t kPlanCount = 8;

  static Result<std::unique_ptr<QwenBf16LinearPlanSet>> Create(
      std::uint64_t tokens, std::uint64_t maximum_workspace_bytes);

  QwenBf16LinearPlanSet(const QwenBf16LinearPlanSet&) = delete;
  QwenBf16LinearPlanSet& operator=(const QwenBf16LinearPlanSet&) = delete;

  Status execute(QwenBf16LinearKind kind, const TensorView& input,
                 const TensorView& weight, const TensorView& output,
                 void* workspace, std::uint64_t workspace_bytes,
                 cudaStream_t stream) const;

  [[nodiscard]] std::uint64_t tokens() const noexcept { return tokens_; }
  [[nodiscard]] std::uint64_t active_logit_rows() const noexcept { return 1; }
  [[nodiscard]] std::uint64_t workspace_bytes() const noexcept {
    return workspace_bytes_;
  }
  [[nodiscard]] Result<std::int32_t> algorithm_id(
      QwenBf16LinearKind kind) const;

 private:
  QwenBf16LinearPlanSet(
      std::uint64_t tokens, std::uint64_t workspace_bytes,
      std::array<std::unique_ptr<GemmPlan>, kPlanCount> plans)
      : tokens_(tokens),
        workspace_bytes_(workspace_bytes),
        plans_(std::move(plans)) {}

  static Result<std::size_t> index(QwenBf16LinearKind kind);

  std::uint64_t tokens_ = 0;
  std::uint64_t workspace_bytes_ = 0;
  std::array<std::unique_ptr<GemmPlan>, kPlanCount> plans_;
};

}  // namespace pih
