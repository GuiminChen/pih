#include "pih/model/deepseek_attention_plan_compute_driver.h"

#include <unordered_set>

namespace pih {

Result<DeepSeekAttentionPlanComputeDriver>
DeepSeekAttentionPlanComputeDriver::Create(
    DeepSeekStageComputeDriver& inner,
    std::span<DeepSeekAttentionSequenceTransaction* const> transactions,
    std::uintptr_t stream) {
  if (transactions.empty() || transactions.size() > 4096 || stream == 0) {
    return Status::InvalidArgument(
        "DeepSeek attention plan transaction resources are invalid");
  }
  std::unordered_set<DeepSeekAttentionSequenceTransaction*> unique;
  for (auto* transaction : transactions) {
    if (transaction == nullptr || !unique.insert(transaction).second) {
      return Status::InvalidArgument(
          "DeepSeek attention plan transactions are null or duplicated");
    }
  }
  DeepSeekAttentionPlanComputeDriver driver;
  driver.inner_ = &inner;
  driver.transactions_.assign(transactions.begin(), transactions.end());
  driver.stream_ = stream;
  return driver;
}

Result<DeepSeekStageComputeStatus>
DeepSeekAttentionPlanComputeDriver::poison(Status status) {
  state_ = State::kPoisoned;
  return status.ok() ? Status::Internal("DeepSeek attention plan poisoned")
                     : status;
}

Status DeepSeekAttentionPlanComputeDriver::launch(
    const DeepSeekPipelinePlanDescriptor& plan,
    const DeepSeekStagePlan& stage) {
  if (state_ != State::kReady || plan.engine_epoch == 0 ||
      plan.plan_sequence == 0) {
    return Status::FailedPrecondition(
        "DeepSeek attention plan compute driver is not launchable");
  }
  manages_state_ = plan.phase != DeepSeekPlanPhase::kDrain;
  if (manages_state_ && plan.sequence_count != transactions_.size()) {
    state_ = State::kPoisoned;
    return Status::InvalidArgument(
        "DeepSeek attention transaction count does not match the plan");
  }
  if (manages_state_) {
    for (const auto* transaction : transactions_) {
      auto status = transaction->validate_begin(stream_);
      if (!status.ok()) {
        state_ = State::kPoisoned;
        return status;
      }
    }
    for (auto* transaction : transactions_) {
      auto status = transaction->begin(stream_);
      if (!status.ok()) {
        compute_failed_ = true;
        return seal_all();
      }
    }
  }
  auto status = inner_->launch(plan, stage);
  if (!status.ok()) {
    compute_failed_ = true;
    if (!manages_state_) {
      state_ = State::kPoisoned;
      return status;
    }
    status = seal_all();
    if (!status.ok()) return status;
    return Status::Ok();
  }
  state_ = State::kCompute;
  return Status::Ok();
}

Status DeepSeekAttentionPlanComputeDriver::seal_all() {
  for (auto* transaction : transactions_) {
    if (transaction->state() ==
        DeepSeekAttentionSequenceTransactionState::kPreparing) {
      auto status = transaction->seal(stream_);
      if (!status.ok()) compute_failed_ = true;
    } else if (transaction->state() ==
               DeepSeekAttentionSequenceTransactionState::kPoisoned) {
      compute_failed_ = true;
    } else {
      compute_failed_ = true;
    }
  }
  state_ = State::kAwaitingState;
  return Status::Ok();
}

Result<DeepSeekStageComputeStatus>
DeepSeekAttentionPlanComputeDriver::resolve_all() {
  bool awaiting = false;
  bool failed = compute_failed_;
  for (auto* transaction : transactions_) {
    if (transaction->state() ==
        DeepSeekAttentionSequenceTransactionState::kAwaitingCompletion) {
      auto result = transaction->poll();
      if (!result.ok()) failed = true;
      else if (*result == DeepSeekExpertAsyncStatus::kInProgress) awaiting = true;
      else if (*result == DeepSeekExpertAsyncStatus::kError) failed = true;
      else if (*result != DeepSeekExpertAsyncStatus::kSuccess) failed = true;
    } else if (transaction->state() ==
               DeepSeekAttentionSequenceTransactionState::kPoisoned) {
      failed = true;
    }
  }
  if (awaiting) return DeepSeekStageComputeStatus::kInProgress;

  if (!failed) {
    for (const auto* transaction : transactions_) {
      auto status = transaction->validate_commit();
      if (!status.ok()) {
        failed = true;
        break;
      }
    }
  }
  if (failed) {
    for (auto* transaction : transactions_) {
      if (transaction->state() ==
          DeepSeekAttentionSequenceTransactionState::kReadyToResolve) {
        (void)transaction->abort();
      }
    }
    state_ = State::kPoisoned;
    return DeepSeekStageComputeStatus::kError;
  }
  for (auto* transaction : transactions_) {
    auto status = transaction->commit();
    if (!status.ok()) return poison(status);
  }
  state_ = State::kComplete;
  return DeepSeekStageComputeStatus::kSuccess;
}

Result<DeepSeekStageComputeStatus>
DeepSeekAttentionPlanComputeDriver::poll() {
  if (state_ == State::kReady || state_ == State::kComplete ||
      state_ == State::kPoisoned) {
    return Status::FailedPrecondition(
        "DeepSeek attention plan compute driver is not pollable");
  }
  if (state_ == State::kCompute) {
    auto result = inner_->poll();
    if (!result.ok()) {
      compute_failed_ = true;
    } else if (*result == DeepSeekStageComputeStatus::kInProgress) {
      return *result;
    } else if (*result == DeepSeekStageComputeStatus::kError) {
      compute_failed_ = true;
    }
    if (!manages_state_) {
      state_ = compute_failed_ ? State::kPoisoned : State::kComplete;
      return compute_failed_ ? DeepSeekStageComputeStatus::kError
                             : DeepSeekStageComputeStatus::kSuccess;
    }
    auto status = seal_all();
    if (!status.ok()) return status;
  }
  return resolve_all();
}

}  // namespace pih
