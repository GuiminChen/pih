#include "pih/model/qwen3_bf16_packed_step_builder.h"

#include <cmath>
#include <limits>

namespace pih {

Result<QwenBf16PackedBuiltStep> QwenBf16PackedStepBuilder::Create(
    const PackedTokenPlan& plan, PackedTokenMetadataView token_metadata,
    QwenBf16PackedKvMetadataView kv_metadata,
    std::uint64_t request_generation, std::int32_t owning_rank,
    std::uint32_t slot_count, float rms_epsilon, float attention_scale,
    const QwenBf16CommandBuffer& commands,
    std::span<const ResolvedKernelFunction> legacy_functions,
    std::span<const ResolvedKernelFunction> packed_functions,
    const QwenBf16WeightResourceSet& weights,
    const QwenBf16StepDeviceOwners& owners,
    std::span<std::byte> pinned_staging_backing,
    std::span<const Qwen3SamplingDescriptor> sampling) {
  const auto sample_count = token_metadata.sample_row_index.size();
  if (request_generation == 0 || owning_rank < 0 ||
      weights.owning_rank() != owning_rank || token_metadata.generation == 0 ||
      token_metadata.generation != kv_metadata.generation ||
      plan.execution_bucket_tokens() != token_metadata.input_token_ids.size() ||
      plan.total_real_tokens() != token_metadata.real_token_count ||
      plan.sequence_count() != kv_metadata.key_token_counts.size() ||
      (!sampling.empty() && sampling.size() != plan.sequence_count()) ||
      sample_count > std::numeric_limits<std::uint32_t>::max() ||
      !std::isfinite(rms_epsilon) || rms_epsilon <= 0.0F ||
      !std::isfinite(attention_scale) || attention_scale <= 0.0F) {
    return Status::InvalidArgument("packed step builder identity is invalid");
  }
  auto staging_layout =
      QwenBf16PackedStepStagingLayout::Create(token_metadata, kv_metadata);
  if (!staging_layout.ok()) return staging_layout.status();
  auto staged = sampling.empty()
                    ? staging_layout->materialize(
                          token_metadata, kv_metadata, pinned_staging_backing)
                    : staging_layout->materialize(
                          token_metadata, kv_metadata, sampling,
                          pinned_staging_backing);
  if (!staged.ok()) return staged;
  auto execution_layout = QwenBf16ExecutionArenaLayout::Create(
      plan.execution_bucket_tokens(), (std::max)(sample_count, std::size_t{1}));
  if (!execution_layout.ok()) return execution_layout.status();
  auto resources = QwenBf16PackedResourceFactory::Create(
      request_generation, owning_rank, slot_count,
      static_cast<std::uint32_t>(sample_count), *execution_layout,
      *staging_layout, owners);
  if (!resources.ok()) return resources.status();
  const QwenBf16PackedKernelContext context{
      request_generation, plan.total_real_tokens(),
      static_cast<std::uint32_t>(plan.sequence_count()), slot_count,
      rms_epsilon, attention_scale};
  auto execution = QwenBf16PackedPreparedExecution::Create(
      commands, legacy_functions, packed_functions, *resources, weights,
      context);
  if (!execution.ok()) return execution.status();
  return QwenBf16PackedBuiltStep(
      std::move(*staging_layout), std::move(*execution_layout),
      std::move(*resources), std::move(*execution));
}

}  // namespace pih
