#pragma once
#include "pih/model/qwen3_int4_engine_bootstrap_plan.h"
#include "pih/model/qwen3_int4_synchronous_backend.h"
namespace pih{
class QwenInt4EngineBackendFactory final{public:
 static Result<QwenInt4SynchronousBackend>Create(
  const QwenInt4EngineBootstrapPlan& bootstrap,
  std::span<const ResolvedKernelFunction> functions,
  const QwenInt4WeightResourceSet& weights,
  QwenInt4SynchronousBackendArenas arenas,
  QwenInt4SynchronousBackendIdentity identity,
  QwenInt4SynchronousBackendDrivers drivers);
};}
