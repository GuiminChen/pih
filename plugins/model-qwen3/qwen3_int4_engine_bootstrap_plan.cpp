#include "pih/model/qwen3_int4_engine_bootstrap_plan.h"
namespace pih{
Result<QwenInt4EngineBootstrapPlan> QwenInt4EngineBootstrapPlan::Create(
 const Qwen3Config& config,std::uint32_t step,std::uint32_t sequence,
 std::uint32_t slots,std::uint64_t workspace,std::uint64_t device,
 std::uint64_t pinned,std::uint64_t reserve,std::uint32_t maximum_batch_sequences){
 auto model=QwenInt4EngineModelPlan::Create(config);if(!model.ok())return model.status();
 return Create(std::move(*model),step,sequence,slots,workspace,device,pinned,
               reserve,maximum_batch_sequences);
}
Result<QwenInt4EngineBootstrapPlan> QwenInt4EngineBootstrapPlan::Create(
 QwenInt4EngineModelPlan model,std::uint32_t step,std::uint32_t sequence,
 std::uint32_t slots,std::uint64_t workspace,std::uint64_t device,
 std::uint64_t pinned,std::uint64_t reserve,std::uint32_t maximum_batch_sequences){
 auto resources=QwenInt4EngineResourcePlan::Create(
  step,sequence,slots,workspace,maximum_batch_sequences);
 if(!resources.ok())return resources.status();
 const Status admitted=resources->admit(device,pinned,reserve);if(!admitted.ok())return admitted;
 return QwenInt4EngineBootstrapPlan(std::move(model),std::move(*resources));
}
Result<QwenInt4SynchronousBackendIdentity>
QwenInt4EngineBootstrapPlan::backend_identity(
 const CudaRuntimeResourceIdentity& runtime,std::uint64_t epoch,
 std::uint64_t timeout_ns)const{
 if(epoch==0||timeout_ns==0||runtime.device_ordinal<0||runtime.rank!=
    static_cast<std::uint32_t>(runtime.device_ordinal)||runtime.worker_generation==0||
    runtime.context==0||runtime.stream==0||runtime.event==0)
  return Status::InvalidArgument("Qwen INT4 runtime identity is invalid");
 return QwenInt4SynchronousBackendIdentity{epoch,1,1,1,timeout_ns,
  runtime.context,runtime.stream,runtime.event,runtime.device_ordinal,
  resources_.runtime().slot_count(),0.000001F,0.0883883476F};
}}
