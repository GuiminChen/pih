#include "pih/model/qwen3_int4_engine_backend_factory.h"
namespace pih{
Result<QwenInt4SynchronousBackend> QwenInt4EngineBackendFactory::Create(
 const QwenInt4EngineBootstrapPlan& bootstrap,
 std::span<const ResolvedKernelFunction> functions,
 const QwenInt4WeightResourceSet& weights,QwenInt4SynchronousBackendArenas arenas,
 QwenInt4SynchronousBackendIdentity identity,QwenInt4SynchronousBackendDrivers drivers){
 if(functions.size()!=QwenInt4KernelBundle::kFunctionCount||
    identity.slot_count==0||
    identity.slot_count>bootstrap.resources().runtime().slot_count()||
    weights.resident_bytes()!=bootstrap.resources().resident_weight_bytes())
  return Status::InvalidArgument("Qwen INT4 engine backend assets drifted");
 return QwenInt4SynchronousBackend::Create(bootstrap.model().commands(),functions,
  weights,arenas,identity,drivers);
}}
