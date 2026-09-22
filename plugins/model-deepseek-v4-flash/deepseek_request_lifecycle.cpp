#include "pih/model/deepseek_request_lifecycle.h"

namespace pih {

Result<DeepSeekRequestLifecycle> DeepSeekRequestLifecycle::Create(
    std::uint64_t request_id, std::uint64_t request_generation) {
  if (request_id == 0 || request_generation == 0) {
    return Status::InvalidArgument(
        "DeepSeek request identity must be nonzero");
  }
  return DeepSeekRequestLifecycle(request_id, request_generation);
}

Status DeepSeekRequestLifecycle::admit() {
  if (state_ != DeepSeekRequestState::kSubmitted) {
    return Status::FailedPrecondition("DeepSeek request cannot be admitted");
  }
  state_ = DeepSeekRequestState::kAdmitted;
  return Status::Ok();
}

Status DeepSeekRequestLifecycle::validate_prepare(
    std::uint64_t plan_sequence) const {
  if (state_ != DeepSeekRequestState::kAdmitted || plan_sequence == 0) {
    return Status::FailedPrecondition("DeepSeek request cannot prepare a plan");
  }
  return Status::Ok();
}

Status DeepSeekRequestLifecycle::prepare(std::uint64_t plan_sequence) {
  const auto status = validate_prepare(plan_sequence);
  if (!status.ok()) return status;
  plan_sequence_ = plan_sequence;
  state_ = DeepSeekRequestState::kPrepared;
  return Status::Ok();
}

Status DeepSeekRequestLifecycle::validate_commit(
    std::uint64_t plan_sequence) const {
  if (state_ != DeepSeekRequestState::kPrepared ||
      plan_sequence != plan_sequence_) {
    return Status::FailedPrecondition("DeepSeek request plan cannot commit");
  }
  return Status::Ok();
}

Status DeepSeekRequestLifecycle::commit(std::uint64_t plan_sequence) {
  const auto status = validate_commit(plan_sequence);
  if (!status.ok()) return status;
  state_ = DeepSeekRequestState::kCommitted;
  return Status::Ok();
}

Status DeepSeekRequestLifecycle::validate_abort_prepare(
    std::uint64_t plan_sequence) const {
  if (state_ != DeepSeekRequestState::kPrepared ||
      plan_sequence != plan_sequence_) {
    return Status::FailedPrecondition(
        "DeepSeek request prepared plan cannot abort");
  }
  return Status::Ok();
}

Status DeepSeekRequestLifecycle::abort_prepare(std::uint64_t plan_sequence) {
  const auto status = validate_abort_prepare(plan_sequence);
  if (!status.ok()) return status;
  plan_sequence_ = 0;
  state_ = DeepSeekRequestState::kAdmitted;
  return Status::Ok();
}

Status DeepSeekRequestLifecycle::validate_cancel(
    std::uint64_t request_generation) const {
  if (request_generation != request_generation_) {
    return Status::InvalidArgument("DeepSeek request generation does not match");
  }
  if (state_ == DeepSeekRequestState::kDraining ||
      state_ == DeepSeekRequestState::kCancelled) {
    return Status::Ok();
  }
  if (terminal()) {
    return Status::FailedPrecondition("DeepSeek terminal request cannot cancel");
  }
  return Status::Ok();
}

Status DeepSeekRequestLifecycle::cancel(std::uint64_t request_generation) {
  const auto status = validate_cancel(request_generation);
  if (!status.ok()) return status;
  if (state_ == DeepSeekRequestState::kDraining ||
      state_ == DeepSeekRequestState::kCancelled) {
    return Status::Ok();
  }
  user_terminal_ = true;
  suppress_output_ = true;
  if (state_ == DeepSeekRequestState::kCommitted) {
    state_ = DeepSeekRequestState::kDraining;
  } else {
    state_ = DeepSeekRequestState::kCancelled;
    backend_drained_ = true;
    plan_sequence_ = 0;
  }
  return Status::Ok();
}

Status DeepSeekRequestLifecycle::validate_backend_complete(
    std::uint64_t plan_sequence) const {
  if ((state_ != DeepSeekRequestState::kCommitted &&
       state_ != DeepSeekRequestState::kDraining) ||
      plan_sequence == 0 || plan_sequence != plan_sequence_) {
    return Status::FailedPrecondition(
        "DeepSeek request completion does not match committed plan");
  }
  return Status::Ok();
}

Status DeepSeekRequestLifecycle::backend_complete(
    std::uint64_t plan_sequence, bool terminal) {
  const auto status = validate_backend_complete(plan_sequence);
  if (!status.ok()) return status;
  backend_drained_ = true;
  plan_sequence_ = 0;
  if (state_ == DeepSeekRequestState::kDraining) {
    state_ = failure_pending_ ? DeepSeekRequestState::kFailed
                              : DeepSeekRequestState::kCancelled;
  } else if (!terminal) {
    state_ = DeepSeekRequestState::kAdmitted;
    backend_drained_ = false;
  } else {
    state_ = DeepSeekRequestState::kCompleted;
    user_terminal_ = true;
  }
  return Status::Ok();
}

Status DeepSeekRequestLifecycle::fail() {
  if (terminal()) {
    return Status::FailedPrecondition("DeepSeek request is already terminal");
  }
  suppress_output_ = true;
  user_terminal_ = true;
  failure_pending_ = true;
  backend_drained_ = state_ != DeepSeekRequestState::kCommitted &&
                     state_ != DeepSeekRequestState::kDraining;
  state_ = backend_drained_ ? DeepSeekRequestState::kFailed
                            : DeepSeekRequestState::kDraining;
  return Status::Ok();
}

Status DeepSeekRequestLifecycle::finish() {
  if (state_ != DeepSeekRequestState::kAdmitted || user_terminal_) {
    return Status::FailedPrecondition(
        "DeepSeek request cannot publish completion in its current state");
  }
  state_ = DeepSeekRequestState::kCompleted;
  user_terminal_ = true;
  backend_drained_ = true;
  return Status::Ok();
}

bool DeepSeekRequestLifecycle::terminal() const noexcept {
  return state_ == DeepSeekRequestState::kCompleted ||
         state_ == DeepSeekRequestState::kCancelled ||
         state_ == DeepSeekRequestState::kFailed;
}

}  // namespace pih
