#pragma once

#include <span>

#include "pih/model/packed_token_plan.h"
#include "pih/model/qwen3_int4_packed_prepared_execution.h"

namespace pih {

class QwenInt4PackedBuiltStep final {
 public:
  QwenInt4PackedBuiltStep(const QwenInt4PackedBuiltStep&)=delete;
  QwenInt4PackedBuiltStep& operator=(const QwenInt4PackedBuiltStep&)=delete;
  QwenInt4PackedBuiltStep(QwenInt4PackedBuiltStep&&) noexcept=default;
  [[nodiscard]] const QwenBf16PackedStepStagingLayout& staging_layout()
      const noexcept{return staging_layout_;}
  [[nodiscard]] const QwenBf16ExecutionArenaLayout& execution_layout()
      const noexcept{return execution_layout_;}
  [[nodiscard]] const QwenBf16PackedResourceSet& resources() const noexcept{
    return resources_;}
  [[nodiscard]] QwenInt4PackedPreparedExecution& compute() noexcept{
    return compute_;}
 private:
  friend class QwenInt4PackedStepBuilder;
  QwenInt4PackedBuiltStep(QwenBf16PackedStepStagingLayout staging,
      QwenBf16ExecutionArenaLayout execution,
      QwenBf16PackedResourceSet resources,
      QwenInt4PackedPreparedExecution compute)
      :staging_layout_(std::move(staging)),execution_layout_(std::move(execution)),
       resources_(std::move(resources)),compute_(std::move(compute)){}
  QwenBf16PackedStepStagingLayout staging_layout_;
  QwenBf16ExecutionArenaLayout execution_layout_;
  QwenBf16PackedResourceSet resources_;
  QwenInt4PackedPreparedExecution compute_;
};

class QwenInt4PackedStepBuilder final {
 public:
  static Result<QwenInt4PackedBuiltStep> Create(
      const PackedTokenPlan& plan,PackedTokenMetadataView token_metadata,
      QwenBf16PackedKvMetadataView kv_metadata,
      std::uint64_t request_generation,std::int32_t owning_rank,
      std::uint32_t slot_count,float rms_epsilon,float attention_scale,
      const QwenBf16CommandBuffer& commands,
      std::span<const ResolvedKernelFunction> int4_functions,
      std::span<const ResolvedKernelFunction> packed_functions,
      const QwenInt4WeightResourceSet& weights,
      const QwenBf16StepDeviceOwners& owners,
      std::span<std::byte> pinned_staging_backing,
      std::span<const Qwen3SamplingDescriptor> sampling={});
};

}  // namespace pih
