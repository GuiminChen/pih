#include "pih/model/deepseek_fixed_state_banks.h"

#include <limits>

namespace pih {

Result<DeepSeekFixedStateBanks> DeepSeekFixedStateBanks::Create(
    DeepSeekExpertArenaSpan first, DeepSeekExpertArenaSpan second,
    std::uintptr_t completion_event,
    DeepSeekFixedStateBankOperations& operations) {
  const auto first_end = static_cast<std::uint64_t>(first.address) + first.bytes;
  const auto second_end = static_cast<std::uint64_t>(second.address) + second.bytes;
  if (first.address == 0 || second.address == 0 || first.bytes == 0 ||
      first.bytes != second.bytes || completion_event == 0 ||
      first_end < first.address || second_end < second.address ||
      !(first_end <= second.address || second_end <= first.address)) {
    return Status::InvalidArgument(
        "DeepSeek fixed state bank resources are invalid");
  }
  DeepSeekFixedStateBanks value;
  value.banks_[0] = first;
  value.banks_[1] = second;
  value.completion_event_ = completion_event;
  value.operations_ = &operations;
  return value;
}

std::uintptr_t DeepSeekFixedStateBanks::committed_address() const noexcept {
  return banks_[committed_bank_].address;
}

std::uintptr_t DeepSeekFixedStateBanks::tentative_address() const noexcept {
  return banks_[committed_bank_ ^ 1U].address;
}

Status DeepSeekFixedStateBanks::prepare(std::uintptr_t stream) {
  auto status = validate_prepare(stream);
  if (!status.ok()) return status;
  status = operations_->copy_d2d_async(
      tentative_address(), committed_address(), banks_[0].bytes, stream);
  if (!status.ok()) {
    state_ = DeepSeekFixedStateTransactionState::kPoisoned;
    return status;
  }
  state_ = DeepSeekFixedStateTransactionState::kPrepared;
  return Status::Ok();
}

Status DeepSeekFixedStateBanks::validate_prepare(
    std::uintptr_t stream) const {
  if (state_ != DeepSeekFixedStateTransactionState::kIdle || stream == 0) {
    return Status::FailedPrecondition(
        "DeepSeek fixed state banks are not preparable");
  }
  return Status::Ok();
}

Status DeepSeekFixedStateBanks::seal(std::uintptr_t stream) {
  if (state_ != DeepSeekFixedStateTransactionState::kPrepared || stream == 0) {
    return Status::FailedPrecondition(
        "DeepSeek fixed state banks are not sealable");
  }
  auto status = operations_->record_event(completion_event_, stream);
  if (!status.ok()) {
    state_ = DeepSeekFixedStateTransactionState::kPoisoned;
    return status;
  }
  state_ = DeepSeekFixedStateTransactionState::kAwaitingCompletion;
  return Status::Ok();
}

Result<DeepSeekExpertAsyncStatus> DeepSeekFixedStateBanks::poll() {
  if (state_ != DeepSeekFixedStateTransactionState::kAwaitingCompletion) {
    return Status::FailedPrecondition(
        "DeepSeek fixed state banks have no completion to poll");
  }
  auto result = operations_->query_event(completion_event_);
  if (!result.ok()) {
    state_ = DeepSeekFixedStateTransactionState::kPoisoned;
    return result.status();
  }
  if (*result == DeepSeekExpertAsyncStatus::kSuccess) {
    state_ = DeepSeekFixedStateTransactionState::kReadyToResolve;
  } else if (*result == DeepSeekExpertAsyncStatus::kError) {
    state_ = DeepSeekFixedStateTransactionState::kPoisoned;
  } else if (*result != DeepSeekExpertAsyncStatus::kInProgress) {
    state_ = DeepSeekFixedStateTransactionState::kPoisoned;
    return Status::Internal(
        "DeepSeek fixed state banks received invalid event state");
  }
  return *result;
}

Status DeepSeekFixedStateBanks::validate_commit() const {
  if (state_ != DeepSeekFixedStateTransactionState::kReadyToResolve) {
    return Status::FailedPrecondition(
        "DeepSeek fixed state banks are not committable");
  }
  if (generation_ == std::numeric_limits<std::uint64_t>::max()) {
    return Status::FailedPrecondition(
        "DeepSeek fixed state generation requires epoch rebuild");
  }
  return Status::Ok();
}

Status DeepSeekFixedStateBanks::commit() {
  auto status = validate_commit();
  if (!status.ok()) {
    if (state_ == DeepSeekFixedStateTransactionState::kReadyToResolve) {
      state_ = DeepSeekFixedStateTransactionState::kPoisoned;
    }
    return status;
  }
  committed_bank_ ^= 1U;
  ++generation_;
  state_ = DeepSeekFixedStateTransactionState::kIdle;
  return Status::Ok();
}

Status DeepSeekFixedStateBanks::cancel_prepared() {
  if (state_ != DeepSeekFixedStateTransactionState::kPrepared) {
    return Status::FailedPrecondition(
        "DeepSeek fixed state banks are not cancellable");
  }
  state_ = DeepSeekFixedStateTransactionState::kIdle;
  return Status::Ok();
}

Status DeepSeekFixedStateBanks::abort() {
  if (state_ != DeepSeekFixedStateTransactionState::kReadyToResolve) {
    return Status::FailedPrecondition(
        "DeepSeek fixed state banks are not abortable");
  }
  state_ = DeepSeekFixedStateTransactionState::kIdle;
  return Status::Ok();
}

}  // namespace pih
