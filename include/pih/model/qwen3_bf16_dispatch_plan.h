#pragma once

#include <cstdint>
#include <utility>

#include "pih/backend/cuda/verified_kernel_launcher.h"
#include "pih/model/qwen3_bf16_kernel_manifest.h"

namespace pih {

class QwenBf16DispatchPlan final {
 public:
  static Result<QwenBf16DispatchPlan> CreateEmbedding(
      const ResolvedKernelFunction& function, const TensorView& table,
      const TensorView& token_ids, const TensorView& output,
      std::int32_t owning_rank);
  static Result<QwenBf16DispatchPlan> CreateElementwise(
      QwenBf16Primitive primitive, const ResolvedKernelFunction& function,
      const TensorView& first, const TensorView& second,
      const TensorView& output, std::int32_t owning_rank);
  static Result<QwenBf16DispatchPlan> CreateRmsNorm(
      const ResolvedKernelFunction& function, const TensorView& input,
      const TensorView& weight, const TensorView& output, float epsilon,
      std::int32_t owning_rank);
  static Result<QwenBf16DispatchPlan> CreateRope(
      const ResolvedKernelFunction& function, const TensorView& input,
      const TensorView& cosine, const TensorView& sine,
      const TensorView& output, std::int32_t owning_rank);
  static Result<QwenBf16DispatchPlan> CreateRopeAngles(
      const ResolvedKernelFunction& function, const TensorView& positions,
      const TensorView& cosine, const TensorView& sine,
      std::int32_t owning_rank);
  static Result<QwenBf16DispatchPlan> CreateGreedyArgmax(
      const ResolvedKernelFunction& function, const TensorView& logits,
      const TensorView& sampled_token, const TensorView& error_flag,
      std::int32_t owning_rank);
  static Result<QwenBf16DispatchPlan> CreateKvAppend(
      const ResolvedKernelFunction& function, const TensorView& key_input,
      const TensorView& value_input, const TensorView& kv_backing,
      const TensorView& slot_states, const TensorView& handles,
      const TensorView& token_offsets, const TensorView& error_flag,
      std::uint32_t owner_sequence_index, std::uint32_t layer,
      std::uint32_t slot_count, std::int32_t owning_rank);
  static Result<QwenBf16DispatchPlan> CreatePagedGqa(
      const ResolvedKernelFunction& function, const TensorView& query,
      const TensorView& output, const TensorView& kv_backing,
      const TensorView& slot_states, const TensorView& handles,
      const TensorView& error_flag, std::uint32_t owner_sequence_index,
      std::uint32_t layer, std::uint64_t query_start_position,
      std::uint32_t key_token_count, float scale, std::uint32_t slot_count,
      std::int32_t owning_rank);

  QwenBf16DispatchPlan(const QwenBf16DispatchPlan&) = delete;
  QwenBf16DispatchPlan& operator=(const QwenBf16DispatchPlan&) = delete;
  QwenBf16DispatchPlan(QwenBf16DispatchPlan&&) noexcept = default;
  QwenBf16DispatchPlan& operator=(QwenBf16DispatchPlan&&) noexcept = default;

  Status submit(KernelLaunchDriver& driver, DriverStreamHandle stream);

  [[nodiscard]] QwenBf16Primitive primitive() const noexcept {
    return primitive_;
  }
  [[nodiscard]] const KernelLaunchGeometry& geometry() const noexcept {
    return geometry_;
  }
  [[nodiscard]] const KernelArgumentPacket& arguments() const noexcept {
    return arguments_;
  }
  [[nodiscard]] bool submitted() const noexcept { return submitted_; }

 private:
  QwenBf16DispatchPlan(QwenBf16Primitive primitive,
                       ResolvedKernelFunction function,
                       KernelLaunchGeometry geometry,
                       KernelArgumentPacket arguments)
      : primitive_(primitive),
        function_(std::move(function)),
        geometry_(geometry),
        arguments_(std::move(arguments)) {}

  QwenBf16Primitive primitive_;
  ResolvedKernelFunction function_;
  KernelLaunchGeometry geometry_;
  KernelArgumentPacket arguments_;
  bool submitted_ = false;
};

}  // namespace pih
