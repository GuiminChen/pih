#include "pih/model/deepseek_attention_terminal_state_drainer.h"

#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
namespace pih {

Result<DeepSeekAttentionTerminalStateDrainer>
DeepSeekAttentionTerminalStateDrainer::Create(
    std::uint32_t sequence,
    DeepSeekAttentionStateReservation& state_reservation,
    DeepSeekRatio4PagePool& ratio4_pool,
    DeepSeekRatio128PagePool& ratio128_pool) {
  if (sequence == 0 || !state_reservation.validate_release().ok()) {
    return Status::FailedPrecondition(
        "DeepSeek terminal state drainer is not creatable");
  }
  DeepSeekAttentionTerminalStateDrainer value;
  value.sequence_ = sequence;
  value.state_reservation_ = &state_reservation;
  value.ratio4_pool_ = &ratio4_pool;
  value.ratio128_pool_ = &ratio128_pool;
  return value;
}

Status DeepSeekAttentionTerminalStateDrainer::drain_committed_sequence_state(
    const DeepSeekDsparkStateCutDecision& decision) {
  if (drained_ || !decision.terminal_drain) {
    return Status::FailedPrecondition(
        "DeepSeek committed sequence state is not drainable");
  }
  auto status = state_reservation_->validate_release();
  if (!status.ok()) return status;
  status = ratio4_pool_->validate_release_published_sequence(sequence_);
  if (!status.ok()) return status;
  status = ratio128_pool_->validate_release_published_sequence(sequence_);
  if (!status.ok()) return status;

  status = ratio4_pool_->release_published_sequence(sequence_);
  if (!status.ok()) return status;
  status = ratio128_pool_->release_published_sequence(sequence_);
  if (!status.ok()) return status;
  status = state_reservation_->release();
  if (!status.ok()) return status;
  drained_ = true;
  return Status::Ok();
}

}  // namespace pih
#endif
