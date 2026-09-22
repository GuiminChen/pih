#include "pih/model/qwen3_int4_packed_prepared_execution.h"

namespace pih {
namespace {

Result<std::size_t> packed_index(QwenBf16PackedPrimitive primitive) {
  const auto value=static_cast<std::size_t>(primitive);
  if(value==0 || value>QwenBf16KernelBundle::kPackedPrimitiveCount)
    return Status::InvalidArgument("unknown packed primitive index");
  return value-1;
}

Result<QwenBf16PackedPrimitive> packed_primitive(
    QwenBf16ExecutionOp operation) {
  switch(operation) {
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
    default:return Status::InvalidArgument("operation has no packed primitive");
  }
}

}  // namespace

Result<QwenInt4PackedPreparedExecution>
QwenInt4PackedPreparedExecution::Create(
    const QwenBf16CommandBuffer& commands,
    std::span<const ResolvedKernelFunction> int4_functions,
    std::span<const ResolvedKernelFunction> packed_functions,
    const QwenBf16PackedResourceSet& resources,
    const QwenInt4WeightResourceSet& weights,
    const QwenBf16PackedKernelContext& context) {
  if(int4_functions.size()!=QwenInt4KernelBundle::kFunctionCount ||
     packed_functions.size()!=QwenBf16KernelBundle::kPackedPrimitiveCount ||
     context.request_generation==0 ||
     context.request_generation!=resources.activations().request_generation() ||
     resources.activations().owning_rank()!=weights.owning_rank()) {
    return Status::InvalidArgument(
        "packed Qwen INT4 execution identity is invalid");
  }
  auto error=resources.activations().view(QwenBf16ActivationSlot::kDeviceError);
  if(!error.ok())return error.status();
  auto prelude=QwenBf16ExecutionPrelude::Create(
      *error,context.request_generation,resources.activations().owning_rank());
  if(!prelude.ok())return prelude.status();
  std::vector<Command> prepared;prepared.reserve(kMaximumCommandCount);
  std::size_t legacy=0,packed=0,int4=0,lm=0;
  for(const auto& command:commands) {
    const bool no_samples=resources.sample_count()==0;
    if(no_samples &&
       ((command.backend==QwenBf16CommandBackend::kKernel &&
         command.execution_step.operation==QwenBf16ExecutionOp::kGreedyArgmax) ||
        (command.backend==QwenBf16CommandBackend::kLinear &&
         command.linear_kind==QwenBf16LinearKind::kLmHead))) continue;
    if(command.backend==QwenBf16CommandBackend::kKernel) {
      if(QwenBf16PackedKernelMaterializer::uses_packed_primitive(command)) {
        auto primitive=packed_primitive(command.execution_step.operation);
        if(!primitive.ok())return primitive.status();
        auto index=packed_index(*primitive);if(!index.ok())return index.status();
        auto plan=QwenBf16PackedKernelMaterializer::CreatePacked(
            command,packed_functions[*index],resources,weights,context);
        if(!plan.ok())return plan.status();
        prepared.emplace_back(std::in_place_type<QwenBf16PackedDispatchPlan>,
                              std::move(*plan));++packed;
      } else {
        const auto index=static_cast<std::size_t>(command.primitive);
        if(index>=QwenInt4KernelBundle::kBf16PrimitiveCount)
          return Status::InvalidArgument("unknown retained BF16 primitive");
        auto plan=QwenBf16PackedKernelMaterializer::CreateLegacy(
            command,int4_functions[index],resources,weights,context);
        if(!plan.ok())return plan.status();
        prepared.emplace_back(std::in_place_type<QwenBf16DispatchPlan>,
                              std::move(*plan));++legacy;
      }
    } else if(command.backend==QwenBf16CommandBackend::kLinear &&
              command.linear_kind!=QwenBf16LinearKind::kLmHead) {
      auto plan=QwenInt4PackedLinearMaterializer::Create(
          command,int4_functions.back(),resources,weights,
          context.request_generation);
      if(!plan.ok())return plan.status();
      prepared.emplace_back(std::in_place_type<QwenInt4DispatchPlan>,
                            std::move(*plan));++int4;
    } else if(command.backend==QwenBf16CommandBackend::kLinear) {
      auto index=packed_index(QwenBf16PackedPrimitive::kSampleHidden);
      if(!index.ok())return index.status();
      auto gather=QwenBf16PackedKernelMaterializer::CreateSampleHidden(
          packed_functions[*index],resources,context);
      if(!gather.ok())return gather.status();
      prepared.emplace_back(std::in_place_type<QwenBf16PackedDispatchPlan>,
                            std::move(*gather));++packed;
      auto binding=QwenInt4PackedLmHeadMaterializer::Create(
          command,resources,weights,context.request_generation);
      if(!binding.ok())return binding.status();
      prepared.emplace_back(std::in_place_type<QwenInt4LmHeadBinding>,
                            std::move(*binding));++lm;
    } else return Status::InvalidArgument("unknown packed INT4 command backend");
  }
  const auto expected=resources.sample_count()==0
      ? QwenBf16CommandBuffer::kCommandCount-2:kMaximumCommandCount;
  if(prepared.size()!=expected || legacy!=253 || int4!=196 ||
     packed!=(resources.sample_count()==0?58:60) ||
     lm!=(resources.sample_count()==0?0:1)) {
    return Status::FailedPrecondition("packed Qwen INT4 command mix drifted");
  }
  return QwenInt4PackedPreparedExecution(std::move(*prelude),
      std::move(prepared),legacy,packed,int4,lm);
}

Status QwenInt4PackedPreparedExecution::run(
    QwenBf16DeviceErrorClearDriver& clear_driver,
    KernelLaunchDriver& kernel_driver,
    QwenInt4LmHeadExecutionDriver& lm_head_driver,
    DriverStreamHandle stream) {
  if(stream==0)return Status::InvalidArgument(
      "packed Qwen INT4 execution requires explicit stream");
  if(state_!=QwenInt4PackedPreparedExecutionState::kPrepared)
    return Status::FailedPrecondition("packed Qwen INT4 execution cannot replay");
  state_=QwenInt4PackedPreparedExecutionState::kRunning;
  auto status=prelude_.submit(clear_driver,stream);
  if(!status.ok()){state_=QwenInt4PackedPreparedExecutionState::kPoisoned;return status;}
  while(next_command_<commands_.size()) {
    auto& command=commands_[next_command_];
    if(auto* plan=std::get_if<QwenBf16DispatchPlan>(&command))
      status=plan->submit(kernel_driver,stream);
    else if(auto* plan=std::get_if<QwenBf16PackedDispatchPlan>(&command))
      status=plan->submit(kernel_driver,stream);
    else if(auto* plan=std::get_if<QwenInt4DispatchPlan>(&command))
      status=plan->submit(kernel_driver,stream);
    else status=lm_head_driver.execute(
        std::get<QwenInt4LmHeadBinding>(command),stream);
    if(!status.ok()){state_=QwenInt4PackedPreparedExecutionState::kPoisoned;return status;}
    ++next_command_;
  }
  state_=QwenInt4PackedPreparedExecutionState::kCompleted;
  return Status::Ok();
}

}  // namespace pih
