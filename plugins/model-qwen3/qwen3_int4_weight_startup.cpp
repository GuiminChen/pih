#include "pih/model/qwen3_int4_weight_startup.h"
namespace pih {
Status QwenInt4WeightStartup::Publish(QwenInt4ResidentWeights& resident,
 TypedCopyDriver& copies,CompletionEventDriver& events,
 CompletionEvidenceProvider& evidence,QwenInt4StartupClock& clock,
 QwenInt4StartupWaiter& waiter){
 if(resident.state()!=QwenInt4ResidentWeightState::kPrepared)
  return Status::FailedPrecondition("Qwen INT4 weights are not startup-ready");
 Status status=resident.submit(copies,events);if(!status.ok())return status;
 for(;;){
  status=resident.poll(events,evidence);if(status.ok())return Status::Ok();
  if(status.code()!=StatusCode::kUnavailable)return status;
  auto now=clock.now_ns();if(!now.ok())return now.status();
  status=resident.expire(*now);if(status.code()!=StatusCode::kUnavailable)return status;
  status=waiter.wait();if(!status.ok())return status;
 }
}
}
