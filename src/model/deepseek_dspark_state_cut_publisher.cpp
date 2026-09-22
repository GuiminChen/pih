#include "pih/model/deepseek_dspark_state_cut_publisher.h"

namespace pih {
Result<DeepSeekDsparkStateCutPublisher>
DeepSeekDsparkStateCutPublisher::Create(
    DeepSeekDsparkStateCutPublishOperations& operations) {
  DeepSeekDsparkStateCutPublisher value;
  value.operations_ = &operations;
  return value;
}
Status DeepSeekDsparkStateCutPublisher::run(Status status) {
  if (!status.ok()) poisoned_ = true;
  return status;
}
Status DeepSeekDsparkStateCutPublisher::publish(
    const DeepSeekDsparkStateCutDecision& decision,
    std::uint32_t world_size,
    std::span<const DeepSeekDsparkStateCutAck> acknowledgements,
    bool engine_poisoned) {
  if (published_ || poisoned_) {
    return Status::FailedPrecondition(
        "DeepSeek DSpark state cut is not publishable");
  }
  auto quorum = verify_deepseek_dspark_state_cut_acks(
      decision, world_size, acknowledgements, engine_poisoned);
  if (!quorum.ok()) return quorum.status();
  auto status = run(operations_->validate_engine_healthy());
  if (!status.ok()) return status;
  status = run(operations_->publish_ledger_and_output(decision));
  if (!status.ok()) return status;
  published_ = true;
  return Status::Ok();
}
}  // namespace pih
