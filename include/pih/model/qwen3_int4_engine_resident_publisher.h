#pragma once
#include "pih/model/qwen3_int4_engine_model_plan.h"
#include "pih/model/qwen3_int4_weight_startup.h"
namespace pih{
class QwenInt4EngineResidentPublisher final{public:
 static Result<QwenInt4ResidentWeights>Publish(
  const QwenInt4EngineModelPlan& model,Allocator& allocator,
  CudaCopyEndpoint canonical_file,std::uint64_t engine_epoch,
  std::int32_t owning_rank,
  std::uint64_t destination_owner_id,std::uintptr_t context,
  DriverStreamHandle stream,DriverEventHandle event,std::uint64_t event_generation,
  std::uint64_t first_plan_id,std::uint64_t submit_ns,std::uint64_t deadline_ns,
  TypedCopyDriver& copies,CompletionEventDriver& events,
  CompletionEvidenceProvider& evidence,QwenInt4StartupClock& clock,
  QwenInt4StartupWaiter& waiter);
};}
