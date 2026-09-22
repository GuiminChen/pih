#include "pih/model/deepseek_nccl_edge_pair.h"

namespace pih {
namespace {

Status validate_pair(const DeepSeekNcclCommunicatorManifest& lower,
                     const DeepSeekNcclCommunicatorManifest& upper) {
  if (lower.engine_epoch != upper.engine_epoch ||
      lower.communicator_generation != upper.communicator_generation ||
      lower.bootstrap_lease_id != upper.bootstrap_lease_id ||
      lower.bootstrap_commitment_id != upper.bootstrap_commitment_id ||
      lower.edge_id != upper.edge_id ||
      lower.config_identity != upper.config_identity ||
      lower.communicator_local_rank != 0 ||
      upper.communicator_local_rank != 1 ||
      lower.local_global_rank != upper.peer_global_rank ||
      lower.peer_global_rank != upper.local_global_rank) {
    return Status::InvalidArgument(
        "DeepSeek NCCL edge endpoints do not share one capability identity");
  }
  return Status::Ok();
}

}  // namespace

Result<DeepSeekNcclEdgePair> DeepSeekNcclEdgePair::Create(
    std::unique_ptr<DeepSeekNcclEdgeEndpoint> lower,
    std::unique_ptr<DeepSeekNcclEdgeEndpoint> upper) {
  if (lower == nullptr || upper == nullptr) {
    return Status::InvalidArgument(
        "DeepSeek NCCL edge pair requires both endpoints");
  }
  auto valid = validate_pair(lower->manifest(), upper->manifest());
  if (!valid.ok()) return valid;
  return DeepSeekNcclEdgePair(std::move(lower), std::move(upper));
}

DeepSeekNcclEdgePair::~DeepSeekNcclEdgePair() {
  if (lower_ == nullptr || upper_ == nullptr) return;
  if (state_ != DeepSeekNcclEdgePairState::kDestroyed &&
      state_ != DeepSeekNcclEdgePairState::kAborted) {
    (void)abort();
  }
}

Status DeepSeekNcclEdgePair::fail(Status cause) noexcept {
  (void)abort();
  return cause.ok() ? Status::Internal("DeepSeek NCCL edge pair failed") : cause;
}

Status DeepSeekNcclEdgePair::reconcile_if_ready() {
  const auto lower_state = lower_->state();
  const auto upper_state = upper_->state();
  if (lower_state == DeepSeekNcclCommunicatorState::kInitInProgress ||
      upper_state == DeepSeekNcclCommunicatorState::kInitInProgress) {
    return Status::Unavailable("DeepSeek NCCL edge init remains in progress");
  }
  if (lower_state != DeepSeekNcclCommunicatorState::kCommunicatorReady ||
      upper_state != DeepSeekNcclCommunicatorState::kCommunicatorReady) {
    return fail(Status::Internal(
        "DeepSeek NCCL edge endpoints did not become ready together"));
  }
  auto status = lower_->reconcile();
  if (!status.ok()) return fail(status);
  status = upper_->reconcile();
  if (!status.ok()) return fail(status);
  if (!lower_->bootstrap_zeroized() || !upper_->bootstrap_zeroized()) {
    return fail(Status::Internal(
        "DeepSeek NCCL bootstrap capability remained live after init"));
  }
  state_ = DeepSeekNcclEdgePairState::kReconciled;
  return Status::Ok();
}

Status DeepSeekNcclEdgePair::begin_init() {
  if (state_ != DeepSeekNcclEdgePairState::kPrepared) {
    return Status::FailedPrecondition("DeepSeek NCCL edge pair is not prepared");
  }
  auto status = lower_->accept_bootstrap(true);
  if (!status.ok()) return fail(status);
  status = upper_->accept_bootstrap(true);
  if (!status.ok()) return fail(status);
  status = lower_->begin_init();
  if (!status.ok()) return fail(status);
  status = upper_->begin_init();
  if (!status.ok()) return fail(status);
  state_ = DeepSeekNcclEdgePairState::kInitInProgress;
  return reconcile_if_ready();
}

Status DeepSeekNcclEdgePair::poll_init() {
  if (state_ != DeepSeekNcclEdgePairState::kInitInProgress) {
    return Status::FailedPrecondition(
        "DeepSeek NCCL edge pair init is not pending");
  }
  if (lower_->state() == DeepSeekNcclCommunicatorState::kInitInProgress) {
    auto status = lower_->poll_init();
    if (!status.ok() && status.code() != StatusCode::kUnavailable) {
      return fail(status);
    }
  }
  if (upper_->state() == DeepSeekNcclCommunicatorState::kInitInProgress) {
    auto status = upper_->poll_init();
    if (!status.ok() && status.code() != StatusCode::kUnavailable) {
      return fail(status);
    }
  }
  return reconcile_if_ready();
}

Status DeepSeekNcclEdgePair::seal(
    const DeepSeekNcclWarmupReceipt& receipt) {
  if (state_ != DeepSeekNcclEdgePairState::kReconciled ||
      receipt.engine_epoch != engine_epoch() ||
      receipt.communicator_generation != communicator_generation() ||
      receipt.edge_id != edge_id()) {
    return Status::FailedPrecondition(
        "DeepSeek NCCL warm-up receipt identity is invalid");
  }
  auto status = lower_->mark_warmed(receipt.minimum_boundary_passed,
                                    receipt.maximum_boundary_passed);
  if (!status.ok()) return fail(status);
  status = upper_->mark_warmed(receipt.minimum_boundary_passed,
                               receipt.maximum_boundary_passed);
  if (!status.ok()) return fail(status);
  status = lower_->seal();
  if (!status.ok()) return fail(status);
  status = upper_->seal();
  if (!status.ok()) return fail(status);
  state_ = DeepSeekNcclEdgePairState::kSealed;
  return Status::Ok();
}

Status DeepSeekNcclEdgePair::begin_finalize() {
  if (state_ != DeepSeekNcclEdgePairState::kSealed) {
    return Status::FailedPrecondition("DeepSeek NCCL edge pair is not sealed");
  }
  auto status = upper_->begin_finalize();
  if (!status.ok()) return fail(status);
  status = lower_->begin_finalize();
  if (!status.ok()) return fail(status);
  state_ = DeepSeekNcclEdgePairState::kFinalizeInProgress;
  if (upper_->state() == DeepSeekNcclCommunicatorState::kFinalizedSuccess &&
      lower_->state() == DeepSeekNcclCommunicatorState::kFinalizedSuccess) {
    return Status::Ok();
  }
  return Status::Unavailable("DeepSeek NCCL edge finalize remains in progress");
}

Status DeepSeekNcclEdgePair::poll_finalize() {
  if (state_ != DeepSeekNcclEdgePairState::kFinalizeInProgress) {
    return Status::FailedPrecondition(
        "DeepSeek NCCL edge finalize is not pending");
  }
  for (auto* endpoint : {upper_.get(), lower_.get()}) {
    if (endpoint->state() ==
        DeepSeekNcclCommunicatorState::kFinalizeInProgress) {
      auto status = endpoint->poll_finalize();
      if (!status.ok() && status.code() != StatusCode::kUnavailable) {
        return fail(status);
      }
    }
  }
  if (upper_->state() != DeepSeekNcclCommunicatorState::kFinalizedSuccess ||
      lower_->state() != DeepSeekNcclCommunicatorState::kFinalizedSuccess) {
    return Status::Unavailable("DeepSeek NCCL edge finalize remains in progress");
  }
  return Status::Ok();
}

Status DeepSeekNcclEdgePair::destroy() {
  if (state_ != DeepSeekNcclEdgePairState::kFinalizeInProgress ||
      upper_->state() != DeepSeekNcclCommunicatorState::kFinalizedSuccess ||
      lower_->state() != DeepSeekNcclCommunicatorState::kFinalizedSuccess) {
    return Status::FailedPrecondition(
        "DeepSeek NCCL edge pair is not cleanly finalized");
  }
  auto status = upper_->destroy();
  if (!status.ok()) return fail(status);
  status = lower_->destroy();
  if (!status.ok()) return fail(status);
  state_ = DeepSeekNcclEdgePairState::kDestroyed;
  return Status::Ok();
}

Status DeepSeekNcclEdgePair::abort() noexcept {
  if (lower_ == nullptr || upper_ == nullptr) return Status::Ok();
  if (state_ == DeepSeekNcclEdgePairState::kAborted) return Status::Ok();
  if (state_ == DeepSeekNcclEdgePairState::kDestroyed) {
    return Status::FailedPrecondition(
        "destroyed DeepSeek NCCL edge pair cannot abort");
  }
  const auto upper_status = upper_->abort();
  const auto lower_status = lower_->abort();
  state_ = DeepSeekNcclEdgePairState::kAborted;
  return !upper_status.ok() ? upper_status : lower_status;
}

}  // namespace pih
