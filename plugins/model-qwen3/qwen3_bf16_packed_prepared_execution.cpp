#include "pih/model/qwen3_bf16_packed_prepared_execution.h"

namespace pih {
namespace {

Result<std::size_t> packed_index(QwenBf16PackedPrimitive primitive) {
  const auto value = static_cast<std::size_t>(primitive);
  if (value == 0 || value > QwenBf16KernelBundle::kPackedPrimitiveCount)
    return Status::InvalidArgument("unknown packed primitive index");
  return value - 1;
}

Result<QwenBf16PackedPrimitive> packed_primitive(
    QwenBf16ExecutionOp operation) {
  switch (operation) {
    case QwenBf16ExecutionOp::kEmbedding:
      return QwenBf16PackedPrimitive::kEmbedding;
    case QwenBf16ExecutionOp::kPrepareRopeAngles:
      return QwenBf16PackedPrimitive::kRopeAngles;
    case QwenBf16ExecutionOp::kKvAppend:
      return QwenBf16PackedPrimitive::kKvAppend;
    case QwenBf16ExecutionOp::kPagedGqa:
      return QwenBf16PackedPrimitive::kPagedGqa;
    case QwenBf16ExecutionOp::kGreedyArgmax:
      return QwenBf16PackedPrimitive::kSampler;
    default:
      return Status::InvalidArgument("operation has no packed primitive");
  }
}

}  // namespace

Result<QwenBf16PackedPreparedExecution>
QwenBf16PackedPreparedExecution::Create(
    const QwenBf16CommandBuffer& commands,
    std::span<const ResolvedKernelFunction> legacy_functions,
    std::span<const ResolvedKernelFunction> packed_functions,
    const QwenBf16PackedResourceSet& resources,
    const QwenBf16WeightResourceSet& weights,
    const QwenBf16PackedKernelContext& context) {
  if (legacy_functions.size() != QwenBf16KernelBundle::kExecutionPrimitiveCount ||
      packed_functions.size() != QwenBf16KernelBundle::kPackedPrimitiveCount ||
      context.request_generation != resources.activations().request_generation()) {
    return Status::InvalidArgument("packed prepared execution identity is invalid");
  }
  auto error = resources.activations().view(QwenBf16ActivationSlot::kDeviceError);
  if (!error.ok()) return error.status();
  auto prelude = QwenBf16ExecutionPrelude::Create(
      *error, context.request_generation, resources.activations().owning_rank());
  if (!prelude.ok()) return prelude.status();
  std::vector<Command> prepared;
  prepared.reserve(kMaximumCommandCount);
  for (const auto& command : commands) {
    if (resources.sample_count() == 0 &&
        ((command.backend == QwenBf16CommandBackend::kKernel &&
          command.execution_step.operation ==
              QwenBf16ExecutionOp::kGreedyArgmax) ||
         (command.backend == QwenBf16CommandBackend::kLinear &&
          command.linear_kind == QwenBf16LinearKind::kLmHead))) {
      continue;
    }
    if (command.backend == QwenBf16CommandBackend::kKernel) {
      if (QwenBf16PackedKernelMaterializer::uses_packed_primitive(command)) {
        auto primitive = packed_primitive(command.execution_step.operation);
        if (!primitive.ok()) return primitive.status();
        auto index = packed_index(*primitive);
        if (!index.ok()) return index.status();
        auto plan = QwenBf16PackedKernelMaterializer::CreatePacked(
            command, packed_functions[*index], resources, weights, context);
        if (!plan.ok()) return plan.status();
        prepared.emplace_back(std::in_place_type<QwenBf16PackedDispatchPlan>,
                              std::move(*plan));
      } else {
        const auto index = static_cast<std::size_t>(command.primitive);
        if (index >= legacy_functions.size())
          return Status::InvalidArgument("unknown legacy kernel primitive");
        auto plan = QwenBf16PackedKernelMaterializer::CreateLegacy(
            command, legacy_functions[index], resources, weights, context);
        if (!plan.ok()) return plan.status();
        prepared.emplace_back(std::in_place_type<QwenBf16DispatchPlan>,
                              std::move(*plan));
      }
    } else if (command.backend == QwenBf16CommandBackend::kLinear) {
      if (command.linear_kind == QwenBf16LinearKind::kLmHead) {
        auto index = packed_index(QwenBf16PackedPrimitive::kSampleHidden);
        if (!index.ok()) return index.status();
        auto gather = QwenBf16PackedKernelMaterializer::CreateSampleHidden(
            packed_functions[*index], resources, context);
        if (!gather.ok()) return gather.status();
        prepared.emplace_back(std::in_place_type<QwenBf16PackedDispatchPlan>,
                              std::move(*gather));
      }
      auto binding = QwenBf16PackedLinearMaterializer::Create(
          command, resources, weights, context.request_generation);
      if (!binding.ok()) return binding.status();
      prepared.emplace_back(std::in_place_type<QwenBf16LinearBinding>,
                            std::move(*binding));
    } else {
      return Status::InvalidArgument("unknown packed command backend");
    }
  }
  const auto expected = resources.sample_count() == 0
                            ? QwenBf16CommandBuffer::kCommandCount - 2
                            : kMaximumCommandCount;
  if (prepared.size() != expected)
    return Status::FailedPrecondition("packed prepared command count drifted");
  return QwenBf16PackedPreparedExecution(std::move(*prelude),
                                         std::move(prepared));
}

Status QwenBf16PackedPreparedExecution::submit(
    QwenBf16DeviceErrorClearDriver& clear_driver,
    KernelLaunchDriver& kernel_driver,
    QwenBf16LinearExecutionDriver& linear_driver,
    DriverStreamHandle stream) {
  if (stream == 0)
    return Status::InvalidArgument("packed execution requires explicit stream");
  if (state_ != QwenBf16PackedPreparedExecutionState::kPrepared)
    return Status::FailedPrecondition("packed execution cannot be replayed");
  state_ = QwenBf16PackedPreparedExecutionState::kRunning;
  auto status = prelude_.submit(clear_driver, stream);
  if (!status.ok()) {
    state_ = QwenBf16PackedPreparedExecutionState::kPoisoned;
    return status;
  }
  while (next_command_ < commands_.size()) {
    auto& command = commands_[next_command_];
    if (auto* legacy = std::get_if<QwenBf16DispatchPlan>(&command)) {
      status = legacy->submit(kernel_driver, stream);
    } else if (auto* packed =
                   std::get_if<QwenBf16PackedDispatchPlan>(&command)) {
      status = packed->submit(kernel_driver, stream);
    } else {
      status = linear_driver.execute(std::get<QwenBf16LinearBinding>(command),
                                     stream);
    }
    if (!status.ok()) {
      state_ = QwenBf16PackedPreparedExecutionState::kPoisoned;
      return status;
    }
    ++next_command_;
  }
  state_ = QwenBf16PackedPreparedExecutionState::kCompleted;
  return Status::Ok();
}

}  // namespace pih
