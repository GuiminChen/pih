#pragma once

#include <cstdint>
#include <span>

#include "pih/model/packed_token_plan.h"
#include "pih/model/qwen3_bf16_packed_prepared_execution.h"
#include "pih/model/qwen3_bf16_packed_resource_factory.h"

namespace pih {

class QwenBf16PackedBuiltStep final {
 public:
  QwenBf16PackedBuiltStep(const QwenBf16PackedBuiltStep&) = delete;
  QwenBf16PackedBuiltStep& operator=(const QwenBf16PackedBuiltStep&) = delete;
  QwenBf16PackedBuiltStep(QwenBf16PackedBuiltStep&&) noexcept = default;

  [[nodiscard]] const QwenBf16PackedStepStagingLayout& staging_layout()
      const noexcept { return staging_layout_; }
  [[nodiscard]] const QwenBf16ExecutionArenaLayout& execution_layout()
      const noexcept { return execution_layout_; }
  [[nodiscard]] const QwenBf16PackedResourceSet& resources() const noexcept {
    return resources_;
  }
  [[nodiscard]] QwenBf16PackedPreparedExecution& compute() noexcept {
    return compute_;
  }

 private:
  friend class QwenBf16PackedStepBuilder;
  QwenBf16PackedBuiltStep(QwenBf16PackedStepStagingLayout staging_layout,
                          QwenBf16ExecutionArenaLayout execution_layout,
                          QwenBf16PackedResourceSet resources,
                          QwenBf16PackedPreparedExecution compute)
      : staging_layout_(std::move(staging_layout)),
        execution_layout_(std::move(execution_layout)),
        resources_(std::move(resources)), compute_(std::move(compute)) {}

  QwenBf16PackedStepStagingLayout staging_layout_;
  QwenBf16ExecutionArenaLayout execution_layout_;
  QwenBf16PackedResourceSet resources_;
  QwenBf16PackedPreparedExecution compute_;
};

class QwenBf16PackedStepBuilder final {
 public:
  static Result<QwenBf16PackedBuiltStep> Create(
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
      std::span<const Qwen3SamplingDescriptor> sampling = {});
};

}  // namespace pih
