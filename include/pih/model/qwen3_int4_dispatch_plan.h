#pragma once

#include <cstdint>

#include "pih/backend/cuda/verified_kernel_launcher.h"
#include "pih/model/qwen3_int4_gemm_plan.h"

namespace pih {

class QwenInt4DispatchPlan final {
 public:
  static Result<QwenInt4DispatchPlan> Create(
      const QwenInt4GemmPlan& gemm,
      const ResolvedKernelFunction& function,
      const TensorView& input, const TensorView& packed_weight,
      const TensorView& scales, const TensorView& output,
      const TensorView& error_flag, std::int32_t owning_rank,
      std::uint64_t activation_generation,
      std::uint64_t weight_generation);

  QwenInt4DispatchPlan(const QwenInt4DispatchPlan&) = delete;
  QwenInt4DispatchPlan& operator=(const QwenInt4DispatchPlan&) = delete;
  QwenInt4DispatchPlan(QwenInt4DispatchPlan&&) noexcept = default;
  QwenInt4DispatchPlan& operator=(QwenInt4DispatchPlan&&) noexcept = default;

  Status submit(KernelLaunchDriver& driver, DriverStreamHandle stream);
  [[nodiscard]] const KernelLaunchGeometry& geometry() const noexcept {
    return geometry_;
  }
  [[nodiscard]] const KernelArgumentPacket& arguments() const noexcept {
    return arguments_;
  }
  [[nodiscard]] bool submitted() const noexcept { return submitted_; }

 private:
  QwenInt4DispatchPlan(ResolvedKernelFunction function,
                       KernelLaunchGeometry geometry,
                       KernelArgumentPacket arguments)
      : function_(std::move(function)), geometry_(geometry),
        arguments_(std::move(arguments)) {}

  ResolvedKernelFunction function_;
  KernelLaunchGeometry geometry_;
  KernelArgumentPacket arguments_;
  bool submitted_ = false;
};

}  // namespace pih
