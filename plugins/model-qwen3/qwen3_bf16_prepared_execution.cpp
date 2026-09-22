#include "pih/model/qwen3_bf16_prepared_execution.h"

#include <utility>

namespace pih {

Result<QwenBf16PreparedExecution> QwenBf16PreparedExecution::Create(
    const QwenBf16CommandBuffer& commands,
    std::span<const ResolvedKernelFunction> functions,
    const QwenBf16ResourceSet& resources,
    const QwenBf16WeightResourceSet& weights,
    const QwenBf16KernelContext& context) {
  if (functions.size() != QwenBf16KernelBundle::kExecutionPrimitiveCount ||
      context.request_generation != resources.request_generation()) {
    return Status::InvalidArgument(
        "Qwen prepared execution identity or function count is invalid");
  }
  auto error = resources.view(QwenBf16ActivationSlot::kDeviceError);
  if (!error.ok()) return error.status();
  auto prelude = QwenBf16ExecutionPrelude::Create(
      *error, context.request_generation, resources.owning_rank());
  if (!prelude.ok()) return prelude.status();

  std::vector<Command> prepared;
  prepared.reserve(QwenBf16CommandBuffer::kCommandCount);
  for (const auto& command : commands) {
    if (command.backend == QwenBf16CommandBackend::kKernel) {
      const auto primitive = static_cast<std::size_t>(command.primitive);
      if (primitive >= functions.size()) {
        return Status::InvalidArgument(
            "Qwen command contains an unknown kernel primitive");
      }
      auto plan = QwenBf16KernelMaterializer::Create(
          command, functions[primitive], resources, weights, context);
      if (!plan.ok()) return plan.status();
      prepared.emplace_back(std::in_place_type<QwenBf16DispatchPlan>,
                            std::move(plan).value());
    } else if (command.backend == QwenBf16CommandBackend::kLinear) {
      auto binding = QwenBf16LinearMaterializer::Create(
          command, resources, weights, context.request_generation);
      if (!binding.ok()) return binding.status();
      prepared.emplace_back(std::in_place_type<QwenBf16LinearBinding>,
                            std::move(binding).value());
    } else {
      return Status::InvalidArgument("Qwen command backend is unknown");
    }
  }
  if (prepared.size() != QwenBf16CommandBuffer::kCommandCount) {
    return Status::FailedPrecondition(
        "Qwen prepared command count differs from frozen schedule");
  }
  return QwenBf16PreparedExecution(std::move(prelude).value(),
                                   std::move(prepared));
}

Status QwenBf16PreparedExecution::run(
    QwenBf16DeviceErrorClearDriver& clear_driver,
    KernelLaunchDriver& kernel_driver,
    QwenBf16LinearExecutionDriver& linear_driver,
    DriverStreamHandle stream) {
  return run_impl(clear_driver, kernel_driver, linear_driver, nullptr, nullptr,
                  stream);
}

Status QwenBf16PreparedExecution::run_instrumented(
    QwenBf16DeviceErrorClearDriver& clear_driver,
    KernelLaunchDriver& kernel_driver,
    QwenBf16LinearExecutionDriver& linear_driver,
    QwenBf16TapSnapshotDriver& snapshot_driver,
    const QwenBf16TapBindingPlan& binding_plan,
    DriverStreamHandle stream) {
  return run_impl(clear_driver, kernel_driver, linear_driver, &snapshot_driver,
                  &binding_plan, stream);
}

Status QwenBf16PreparedExecution::run_instrumented_and_record(
    QwenBf16DeviceErrorClearDriver& clear_driver,
    KernelLaunchDriver& kernel_driver,
    QwenBf16LinearExecutionDriver& linear_driver,
    QwenBf16TapSnapshotDriver& snapshot_driver,
    const QwenBf16TapBindingPlan& binding_plan,
    QwenBf16TapSnapshotFrontierRecorder& frontier_recorder,
    CompletionEventDriver& event_driver,
    DriverStreamHandle stream,
    std::uint64_t submit_ns,
    std::uint64_t deadline_ns) {
  Status status = run_instrumented(clear_driver, kernel_driver, linear_driver,
                                   snapshot_driver, binding_plan, stream);
  if (!status.ok()) return status;
  status = frontier_recorder.record_snapshot(event_driver, submit_ns,
                                             deadline_ns);
  if (!status.ok()) state_ = QwenBf16PreparedExecutionState::kPoisoned;
  return status;
}

Status QwenBf16PreparedExecution::run_impl(
    QwenBf16DeviceErrorClearDriver& clear_driver,
    KernelLaunchDriver& kernel_driver,
    QwenBf16LinearExecutionDriver& linear_driver,
    QwenBf16TapSnapshotDriver* snapshot_driver,
    const QwenBf16TapBindingPlan* binding_plan,
    DriverStreamHandle stream) {
  if (stream == 0) {
    return Status::InvalidArgument(
        "Qwen prepared execution requires an explicit stream");
  }
  if (state_ != QwenBf16PreparedExecutionState::kPrepared) {
    return Status::FailedPrecondition(
        "Qwen prepared execution cannot be replayed");
  }
  state_ = QwenBf16PreparedExecutionState::kRunning;
  Status status = prelude_.submit(clear_driver, stream);
  if (!status.ok()) {
    state_ = QwenBf16PreparedExecutionState::kPoisoned;
    return status;
  }
  while (next_command_ < commands_.size()) {
    auto& command = commands_[next_command_];
    if (auto* kernel = std::get_if<QwenBf16DispatchPlan>(&command)) {
      status = kernel->submit(kernel_driver, stream);
    } else {
      status = linear_driver.execute(
          std::get<QwenBf16LinearBinding>(command), stream);
    }
    if (!status.ok()) {
      state_ = QwenBf16PreparedExecutionState::kPoisoned;
      return status;
    }
    if (snapshot_driver != nullptr) {
      for (const auto binding_index :
           binding_plan->bindings_after_command(next_command_)) {
        status = snapshot_driver->snapshot((*binding_plan)[binding_index],
                                           stream);
        if (!status.ok()) {
          state_ = QwenBf16PreparedExecutionState::kPoisoned;
          return status;
        }
      }
    }
    ++next_command_;
  }
  state_ = QwenBf16PreparedExecutionState::kCompleted;
  return Status::Ok();
}

}  // namespace pih
