#include "pih/model/qwen3_int4_engine_resident_publisher.h"
namespace pih{
Result<QwenInt4ResidentWeights> QwenInt4EngineResidentPublisher::Publish(
 const QwenInt4EngineModelPlan& model,Allocator& allocator,CudaCopyEndpoint canonical,
 std::uint64_t epoch,std::int32_t rank,std::uint64_t destination,std::uintptr_t context,
 DriverStreamHandle stream,DriverEventHandle event,std::uint64_t event_generation,
 std::uint64_t first_plan,std::uint64_t submit_ns,std::uint64_t deadline_ns,
 TypedCopyDriver& copies,CompletionEventDriver& events,
 CompletionEvidenceProvider& evidence,QwenInt4StartupClock& clock,QwenInt4StartupWaiter& waiter){
 auto resident=QwenInt4ResidentWeights::Create(model.layout(),model.ledger(),epoch,allocator,
  canonical,rank,destination,context,stream,event,event_generation,first_plan,
  submit_ns,deadline_ns);
 if(!resident.ok())return resident.status();
 const Status published=QwenInt4WeightStartup::Publish(*resident,copies,events,evidence,clock,waiter);
 if(!published.ok())return published;
 if(!resident->resources().ok())return Status::Internal("Qwen INT4 resident publication drifted");
 return std::move(*resident);
}}
