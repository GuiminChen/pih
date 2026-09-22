#pragma once

#include <cstdint>
#include <span>

#include "pih/model/qwen3_bf16_step_resource_factory.h"
#include "pih/model/qwen3_bf16_step_transaction.h"

namespace pih {

class QwenBf16BuiltStep final {
 public:
  QwenBf16BuiltStep(const QwenBf16BuiltStep&) = delete;
  QwenBf16BuiltStep& operator=(const QwenBf16BuiltStep&) = delete;
  QwenBf16BuiltStep(QwenBf16BuiltStep&&) noexcept = default;
  QwenBf16BuiltStep& operator=(QwenBf16BuiltStep&&) = delete;

  [[nodiscard]] const QwenBf16StepStagingLayout& staging_layout()
      const noexcept {
    return staging_layout_;
  }
  [[nodiscard]] const QwenBf16ExecutionArenaLayout& execution_layout()
      const noexcept {
    return execution_layout_;
  }
  [[nodiscard]] QwenBf16PreparedStepCompute& compute() noexcept {
    return compute_;
  }
  [[nodiscard]] const QwenBf16ResourceSet& resources() const noexcept {
    return resources_;
  }

 private:
  friend class QwenBf16StepBuilder;
  QwenBf16BuiltStep(QwenBf16StepStagingLayout staging_layout,
                    QwenBf16ExecutionArenaLayout execution_layout,
                    QwenBf16ResourceSet resources,
                    QwenBf16PreparedStepCompute compute)
      : staging_layout_(std::move(staging_layout)),
        execution_layout_(std::move(execution_layout)),
        resources_(std::move(resources)),
        compute_(std::move(compute)) {}

  QwenBf16StepStagingLayout staging_layout_;
  QwenBf16ExecutionArenaLayout execution_layout_;
  QwenBf16ResourceSet resources_;
  QwenBf16PreparedStepCompute compute_;
};

class QwenBf16StepBuilder final {
 public:
  static Result<QwenBf16BuiltStep> Create(
      std::span<const std::int64_t> tokens, std::uint64_t first_position,
      const QwenKvBlockTable& block_table,
      const QwenKvAppendPlan& append_plan,
      std::uint64_t request_generation, std::int32_t owning_rank,
      std::uint32_t slot_count, float rms_epsilon, float attention_scale,
      const QwenBf16CommandBuffer& commands,
      std::span<const ResolvedKernelFunction> functions,
      const QwenBf16WeightResourceSet& weights,
      const QwenBf16StepDeviceOwners& owners,
      std::span<std::byte> pinned_staging_backing);
};

}  // namespace pih
