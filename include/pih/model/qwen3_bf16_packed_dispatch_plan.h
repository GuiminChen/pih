#pragma once

#include <cstdint>

#include "pih/backend/cuda/verified_kernel_launcher.h"
#include "pih/model/qwen3_bf16_packed_kernel_manifest.h"

namespace pih {

class QwenBf16PackedDispatchPlan final {
 public:
  static Result<QwenBf16PackedDispatchPlan> CreateEmbedding(
      const ResolvedKernelFunction& function, const TensorView& table,
      const TensorView& token_ids, const TensorView& output,
      std::uint64_t real_token_count, std::int32_t owning_rank);
  static Result<QwenBf16PackedDispatchPlan> CreateRopeAngles(
      const ResolvedKernelFunction& function, const TensorView& positions,
      const TensorView& cosine, const TensorView& sine,
      std::uint64_t real_token_count, std::int32_t owning_rank);
  static Result<QwenBf16PackedDispatchPlan> CreateKvAppend(
      const ResolvedKernelFunction& function, const TensorView& key_input,
      const TensorView& value_input, const TensorView& kv_backing,
      const TensorView& slot_states, const TensorView& handles,
      const TensorView& token_offsets, const TensorView& request_index,
      const TensorView& owner_sequence_indices, const TensorView& error_flag,
      std::uint32_t layer, std::uint64_t token_count,
      std::uint32_t sequence_count, std::uint32_t slot_count,
      std::int32_t owning_rank);
  static Result<QwenBf16PackedDispatchPlan> CreatePagedGqa(
      const ResolvedKernelFunction& function, const TensorView& query,
      const TensorView& output, const TensorView& kv_backing,
      const TensorView& slot_states, const TensorView& handles,
      const TensorView& visible_handle_offsets,
      const TensorView& request_index, const TensorView& query_start_offsets,
      const TensorView& key_token_counts,
      const TensorView& owner_sequence_indices, const TensorView& error_flag,
      std::uint32_t layer, std::uint32_t query_count,
      std::uint32_t sequence_count, float scale, std::uint32_t slot_count,
      std::int32_t owning_rank);
  static Result<QwenBf16PackedDispatchPlan> CreateSampleHidden(
      const ResolvedKernelFunction& function, const TensorView& input,
      const TensorView& sample_rows, const TensorView& output,
      const TensorView& error_flag, std::uint32_t sample_count,
      std::uint32_t packed_token_count, std::int32_t owning_rank);
  static Result<QwenBf16PackedDispatchPlan> CreateGreedyArgmax(
      const ResolvedKernelFunction& function, const TensorView& logits,
      const TensorView& sampled_token_ids, const TensorView& error_flag,
      std::uint32_t sample_count, std::int32_t owning_rank);
  static Result<QwenBf16PackedDispatchPlan> CreateSampler(
      const ResolvedKernelFunction& function, const TensorView& logits,
      const TensorView& sampling_descriptors,
      const TensorView& sample_sequence_indices,
      const TensorView& workspace_ids, const TensorView& sampled_token_ids,
      const TensorView& selected_logprobs, const TensorView& rng_words,
      const TensorView& top_token_ids, const TensorView& top_logprobs,
      const TensorView& top_counts, const TensorView& error_flag,
      std::uint32_t sample_count, std::uint32_t sequence_count,
      std::int32_t owning_rank);

  QwenBf16PackedDispatchPlan(const QwenBf16PackedDispatchPlan&) = delete;
  QwenBf16PackedDispatchPlan& operator=(const QwenBf16PackedDispatchPlan&) =
      delete;
  QwenBf16PackedDispatchPlan(QwenBf16PackedDispatchPlan&&) noexcept = default;

  Status submit(KernelLaunchDriver& driver, DriverStreamHandle stream);
  [[nodiscard]] QwenBf16PackedPrimitive primitive() const noexcept {
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
  QwenBf16PackedDispatchPlan(QwenBf16PackedPrimitive primitive,
                             ResolvedKernelFunction function,
                             KernelLaunchGeometry geometry,
                             KernelArgumentPacket arguments)
      : primitive_(primitive), function_(std::move(function)),
        geometry_(geometry), arguments_(std::move(arguments)) {}

  QwenBf16PackedPrimitive primitive_;
  ResolvedKernelFunction function_;
  KernelLaunchGeometry geometry_;
  KernelArgumentPacket arguments_;
  bool submitted_ = false;
};

}  // namespace pih
