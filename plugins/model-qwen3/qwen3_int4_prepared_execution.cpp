#include "pih/model/qwen3_int4_prepared_execution.h"

namespace pih {

Result<QwenInt4PreparedExecution> QwenInt4PreparedExecution::Create(
    const QwenBf16CommandBuffer& commands,
    std::span<const ResolvedKernelFunction> functions,
    const QwenBf16ResourceSet& activations,
    const QwenInt4WeightResourceSet& weights,
    const QwenBf16KernelContext& context) {
  if(functions.size()!=QwenInt4KernelBundle::kFunctionCount ||
     context.request_generation==0 ||
     context.request_generation!=activations.request_generation() ||
     activations.owning_rank()!=weights.owning_rank())
    return Status::InvalidArgument("Qwen INT4 execution identity is invalid");
  auto error=activations.view(QwenBf16ActivationSlot::kDeviceError);
  if(!error.ok())return error.status();
  auto prelude=QwenBf16ExecutionPrelude::Create(
      *error,context.request_generation,activations.owning_rank());
  if(!prelude.ok())return prelude.status();
  std::vector<Command> prepared; prepared.reserve(commands.size());
  std::size_t bf16=0,int4=0,lm=0;
  for(const auto& command:commands) {
    if(command.backend==QwenBf16CommandBackend::kKernel) {
      const auto primitive=static_cast<std::size_t>(command.primitive);
      if(primitive>=QwenInt4KernelBundle::kBf16PrimitiveCount)
        return Status::InvalidArgument("Qwen INT4 kernel primitive is invalid");
      auto plan=QwenBf16KernelMaterializer::Create(
          command,functions[primitive],activations,weights,context);
      if(!plan.ok())return plan.status();
      prepared.emplace_back(std::in_place_type<QwenBf16DispatchPlan>,
                            std::move(*plan)); ++bf16;
    } else if(command.backend==QwenBf16CommandBackend::kLinear &&
              command.linear_kind!=QwenBf16LinearKind::kLmHead) {
      auto plan=QwenInt4LinearMaterializer::Create(
          command,functions.back(),activations,weights,
          context.request_generation);
      if(!plan.ok())return plan.status();
      prepared.emplace_back(std::in_place_type<QwenInt4DispatchPlan>,
                            std::move(*plan)); ++int4;
    } else if(command.backend==QwenBf16CommandBackend::kLinear) {
      auto binding=QwenInt4LmHeadMaterializer::Create(
          command,activations,weights,context.request_generation);
      if(!binding.ok())return binding.status();
      prepared.emplace_back(std::in_place_type<QwenInt4LmHeadBinding>,
                            std::move(*binding)); ++lm;
    } else return Status::InvalidArgument("Qwen INT4 command backend is invalid");
  }
  if(prepared.size()!=QwenBf16CommandBuffer::kCommandCount || bf16!=312 ||
     int4!=196 || lm!=1)
    return Status::FailedPrecondition("Qwen INT4 command mix drifted");
  return QwenInt4PreparedExecution(std::move(*prelude),std::move(prepared),
                                   bf16,int4,lm);
}

Status QwenInt4PreparedExecution::run(
    QwenBf16DeviceErrorClearDriver& clear_driver,
    KernelLaunchDriver& kernel_driver,
    QwenInt4LmHeadExecutionDriver& lm_head_driver,
    DriverStreamHandle stream) {
  if(stream==0)return Status::InvalidArgument("Qwen INT4 execution requires stream");
  if(state_!=QwenInt4PreparedExecutionState::kPrepared)
    return Status::FailedPrecondition("Qwen INT4 execution cannot replay");
  state_=QwenInt4PreparedExecutionState::kRunning;
  auto status=prelude_.submit(clear_driver,stream);
  if(!status.ok()){state_=QwenInt4PreparedExecutionState::kPoisoned;return status;}
  while(next_command_<commands_.size()) {
    auto& command=commands_[next_command_];
    if(auto* plan=std::get_if<QwenBf16DispatchPlan>(&command))
      status=plan->submit(kernel_driver,stream);
    else if(auto* plan=std::get_if<QwenInt4DispatchPlan>(&command))
      status=plan->submit(kernel_driver,stream);
    else status=lm_head_driver.execute(std::get<QwenInt4LmHeadBinding>(command),stream);
    if(!status.ok()){state_=QwenInt4PreparedExecutionState::kPoisoned;return status;}
    ++next_command_;
  }
  state_=QwenInt4PreparedExecutionState::kCompleted;
  return Status::Ok();
}

}  // namespace pih
