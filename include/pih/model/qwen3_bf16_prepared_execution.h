#pragma once

#include <cstddef>
#include <span>
#include <variant>
#include <vector>

#include "pih/backend/cuda/completion_event_slot.h"
#include "pih/model/qwen3_bf16_execution_prelude.h"
#include "pih/model/qwen3_bf16_kernel_bundle.h"
#include "pih/model/qwen3_bf16_kernel_materializer.h"
#include "pih/model/qwen3_bf16_linear_materializer.h"
#include "pih/model/qwen3_bf16_tap_binding.h"

namespace pih {

enum class QwenBf16PreparedExecutionState : std::uint8_t {
  kPrepared,
  kRunning,
  kCompleted,
  kPoisoned,
};

class QwenBf16LinearExecutionDriver {
 public:
  virtual ~QwenBf16LinearExecutionDriver() = default;
  virtual Status execute(const QwenBf16LinearBinding& binding,
                         DriverStreamHandle stream) = 0;
};

class QwenBf16TapSnapshotDriver {
 public:
  virtual ~QwenBf16TapSnapshotDriver() = default;
  virtual Status snapshot(const QwenBf16TapBinding& binding,
                          DriverStreamHandle stream) = 0;
};

class QwenBf16TapSnapshotFrontierRecorder {
 public:
  virtual ~QwenBf16TapSnapshotFrontierRecorder() = default;
  virtual Status record_snapshot(CompletionEventDriver& event_driver,
                                 std::uint64_t submit_ns,
                                 std::uint64_t deadline_ns) = 0;
};

class QwenBf16PreparedExecution final {
 public:
  using Command =
      std::variant<QwenBf16DispatchPlan, QwenBf16LinearBinding>;

  static Result<QwenBf16PreparedExecution> Create(
      const QwenBf16CommandBuffer& commands,
      std::span<const ResolvedKernelFunction> functions,
      const QwenBf16ResourceSet& resources,
      const QwenBf16WeightResourceSet& weights,
      const QwenBf16KernelContext& context);

  QwenBf16PreparedExecution(const QwenBf16PreparedExecution&) = delete;
  QwenBf16PreparedExecution& operator=(const QwenBf16PreparedExecution&) =
      delete;
  QwenBf16PreparedExecution(QwenBf16PreparedExecution&&) noexcept = default;
  QwenBf16PreparedExecution& operator=(QwenBf16PreparedExecution&&) noexcept =
      default;

  Status run(QwenBf16DeviceErrorClearDriver& clear_driver,
             KernelLaunchDriver& kernel_driver,
             QwenBf16LinearExecutionDriver& linear_driver,
             DriverStreamHandle stream);

  Status run_instrumented(
      QwenBf16DeviceErrorClearDriver& clear_driver,
      KernelLaunchDriver& kernel_driver,
      QwenBf16LinearExecutionDriver& linear_driver,
      QwenBf16TapSnapshotDriver& snapshot_driver,
      const QwenBf16TapBindingPlan& binding_plan,
      DriverStreamHandle stream);

  Status run_instrumented_and_record(
      QwenBf16DeviceErrorClearDriver& clear_driver,
      KernelLaunchDriver& kernel_driver,
      QwenBf16LinearExecutionDriver& linear_driver,
      QwenBf16TapSnapshotDriver& snapshot_driver,
      const QwenBf16TapBindingPlan& binding_plan,
      QwenBf16TapSnapshotFrontierRecorder& frontier_recorder,
      CompletionEventDriver& event_driver,
      DriverStreamHandle stream,
      std::uint64_t submit_ns,
      std::uint64_t deadline_ns);

  [[nodiscard]] std::size_t size() const noexcept { return commands_.size(); }
  [[nodiscard]] std::size_t next_command() const noexcept {
    return next_command_;
  }
  [[nodiscard]] QwenBf16PreparedExecutionState state() const noexcept {
    return state_;
  }

 private:
  QwenBf16PreparedExecution(QwenBf16ExecutionPrelude prelude,
                            std::vector<Command> commands)
      : prelude_(std::move(prelude)), commands_(std::move(commands)) {}

  QwenBf16ExecutionPrelude prelude_;
  std::vector<Command> commands_;
  std::size_t next_command_ = 0;
  QwenBf16PreparedExecutionState state_ =
      QwenBf16PreparedExecutionState::kPrepared;

  Status run_impl(QwenBf16DeviceErrorClearDriver& clear_driver,
                  KernelLaunchDriver& kernel_driver,
                  QwenBf16LinearExecutionDriver& linear_driver,
                  QwenBf16TapSnapshotDriver* snapshot_driver,
                  const QwenBf16TapBindingPlan* binding_plan,
                  DriverStreamHandle stream);
};

}  // namespace pih
