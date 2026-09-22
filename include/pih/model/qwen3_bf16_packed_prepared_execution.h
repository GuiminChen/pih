#pragma once

#include <span>
#include <variant>
#include <vector>

#include "pih/model/qwen3_bf16_packed_kernel_materializer.h"
#include "pih/model/qwen3_bf16_packed_linear_materializer.h"
#include "pih/model/qwen3_bf16_step_transaction.h"

namespace pih {

enum class QwenBf16PackedPreparedExecutionState : std::uint8_t {
  kPrepared = 1,
  kRunning = 2,
  kCompleted = 3,
  kPoisoned = 4,
};

class QwenBf16PackedPreparedExecution final : public QwenBf16StepCompute {
 public:
  static constexpr std::size_t kMaximumCommandCount =
      QwenBf16CommandBuffer::kCommandCount + 1;
  using Command = std::variant<QwenBf16DispatchPlan,
                               QwenBf16PackedDispatchPlan,
                               QwenBf16LinearBinding>;

  static Result<QwenBf16PackedPreparedExecution> Create(
      const QwenBf16CommandBuffer& commands,
      std::span<const ResolvedKernelFunction> legacy_functions,
      std::span<const ResolvedKernelFunction> packed_functions,
      const QwenBf16PackedResourceSet& resources,
      const QwenBf16WeightResourceSet& weights,
      const QwenBf16PackedKernelContext& context);

  QwenBf16PackedPreparedExecution(
      const QwenBf16PackedPreparedExecution&) = delete;
  QwenBf16PackedPreparedExecution& operator=(
      const QwenBf16PackedPreparedExecution&) = delete;
  QwenBf16PackedPreparedExecution(
      QwenBf16PackedPreparedExecution&&) noexcept = default;

  Status submit(QwenBf16DeviceErrorClearDriver& clear_driver,
                KernelLaunchDriver& kernel_driver,
                QwenBf16LinearExecutionDriver& linear_driver,
                DriverStreamHandle stream) override;

  [[nodiscard]] std::size_t size() const noexcept { return commands_.size(); }
  [[nodiscard]] std::size_t next_command() const noexcept {
    return next_command_;
  }
  [[nodiscard]] QwenBf16PackedPreparedExecutionState state() const noexcept {
    return state_;
  }

 private:
  QwenBf16PackedPreparedExecution(QwenBf16ExecutionPrelude prelude,
                                  std::vector<Command> commands)
      : prelude_(std::move(prelude)), commands_(std::move(commands)) {}

  QwenBf16ExecutionPrelude prelude_;
  std::vector<Command> commands_;
  std::size_t next_command_ = 0;
  QwenBf16PackedPreparedExecutionState state_ =
      QwenBf16PackedPreparedExecutionState::kPrepared;
};

}  // namespace pih
