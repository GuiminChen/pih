#pragma once
#include "pih/model/qwen3_int4_engine_bootstrap_plan.h"
namespace pih{
class NvidiaQwenInt4CapacityProbe final{public:
 static Result<QwenInt4EngineBootstrapPlan> Admit(
  const Qwen3Config& config,std::int32_t device_ordinal,
  std::uint32_t maximum_step_tokens,std::uint32_t maximum_sequence_tokens,
  std::uint32_t slot_count,std::uint64_t linear_workspace_bytes,
  std::uint64_t available_pinned_bytes,std::uint64_t device_reserve_bytes,
  std::uint32_t maximum_batch_sequences=1);
 static Result<QwenInt4EngineBootstrapPlan> Admit(
  QwenInt4EngineModelPlan model,std::int32_t device_ordinal,
  std::uint32_t maximum_step_tokens,std::uint32_t maximum_sequence_tokens,
  std::uint32_t slot_count,std::uint64_t linear_workspace_bytes,
  std::uint64_t available_pinned_bytes,std::uint64_t device_reserve_bytes,
  std::uint32_t maximum_batch_sequences=1);
};}
