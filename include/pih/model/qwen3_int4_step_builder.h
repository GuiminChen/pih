#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "pih/model/qwen3_bf16_step_resource_factory.h"
#include "pih/model/qwen3_int4_prepared_execution.h"

namespace pih {

class QwenInt4BuiltStep final {
 public:
  QwenInt4BuiltStep(const QwenInt4BuiltStep&) = delete;
  QwenInt4BuiltStep& operator=(const QwenInt4BuiltStep&) = delete;
  QwenInt4BuiltStep(QwenInt4BuiltStep&&) noexcept = default;
  QwenInt4BuiltStep& operator=(QwenInt4BuiltStep&&) = delete;

  [[nodiscard]] const QwenBf16StepStagingLayout& staging_layout()
      const noexcept { return staging_layout_; }
  [[nodiscard]] const QwenBf16ExecutionArenaLayout& execution_layout()
      const noexcept { return execution_layout_; }
  [[nodiscard]] const QwenBf16ResourceSet& resources() const noexcept {
    return resources_;
  }
  [[nodiscard]] QwenInt4PreparedExecution& execution() noexcept {
    return execution_;
  }

 private:
  friend class QwenInt4StepBuilder;
  QwenInt4BuiltStep(QwenBf16StepStagingLayout staging_layout,
                    QwenBf16ExecutionArenaLayout execution_layout,
                    QwenBf16ResourceSet resources,
                    QwenInt4PreparedExecution execution)
      : staging_layout_(std::move(staging_layout)),
        execution_layout_(std::move(execution_layout)),
        resources_(std::move(resources)),
        execution_(std::move(execution)) {}

  QwenBf16StepStagingLayout staging_layout_;
  QwenBf16ExecutionArenaLayout execution_layout_;
  QwenBf16ResourceSet resources_;
  QwenInt4PreparedExecution execution_;
};

class QwenInt4StepBuilder final {
 public:
  static Result<QwenInt4BuiltStep> Create(
      std::span<const std::int64_t> tokens, std::uint64_t first_position,
      const QwenKvBlockTable& block_table,
      const QwenKvAppendPlan& append_plan,
      std::uint64_t request_generation, std::int32_t owning_rank,
      std::uint32_t slot_count, float rms_epsilon, float attention_scale,
      const QwenBf16CommandBuffer& commands,
      std::span<const ResolvedKernelFunction> functions,
      const QwenInt4WeightResourceSet& weights,
      const QwenBf16StepDeviceOwners& owners,
      std::span<std::byte> pinned_staging_backing);
};

}  // namespace pih
