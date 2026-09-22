#include "pih/model/qwen3_numerical_tap_transfer.h"

namespace pih {

Result<QwenNumericalTapTransfer> QwenNumericalTapTransfer::Create(
    std::size_t capture_index, std::uint64_t capture_generation,
    std::uint64_t bytes, std::uint64_t producer_plan_generation,
    CudaTypedCopyPlan snapshot_copy, CudaTypedCopyPlan host_copy) {
  const bool fixed_identity =
      capture_generation != 0 && bytes != 0 &&
      producer_plan_generation != 0 && snapshot_copy.bytes() == bytes &&
      host_copy.bytes() == bytes &&
      snapshot_copy.purpose() == CudaCopyPurpose::kSameRankMove &&
      host_copy.purpose() == CudaCopyPurpose::kDiagnostic &&
      snapshot_copy.kind() == CudaCopyKind::kDeviceToDevice &&
      host_copy.kind() == CudaCopyKind::kDeviceToHost &&
      snapshot_copy.destination_owner_id() == host_copy.source_owner_id() &&
      snapshot_copy.destination_generation() ==
          host_copy.source_generation() &&
      snapshot_copy.destination_address() == host_copy.source_address() &&
      snapshot_copy.context_identity() == host_copy.context_identity() &&
      snapshot_copy.completion_event_generation() !=
          host_copy.completion_event_generation();
  if (!fixed_identity) {
    return Status::InvalidArgument(
        "Qwen numerical tap transfer chain is invalid");
  }
  return QwenNumericalTapTransfer(
      capture_index, capture_generation, bytes, producer_plan_generation,
      std::move(snapshot_copy), std::move(host_copy));
}

Status QwenNumericalTapTransfer::poison(const char* message) {
  state_ = QwenNumericalTapTransferState::kPoisoned;
  return Status::FailedPrecondition(message);
}

Status QwenNumericalTapTransfer::complete_producer(
    const CudaCompletionFrontier& frontier) {
  if (state_ != QwenNumericalTapTransferState::kProducerPending ||
      !frontier.publication_authorized() ||
      frontier.key().plan_generation != producer_plan_generation_) {
    return poison("Qwen tap producer frontier is invalid");
  }
  state_ = QwenNumericalTapTransferState::kSnapshotReady;
  return Status::Ok();
}

Status QwenNumericalTapTransfer::submit_inline_snapshot(
    TypedCopyDriver& driver, std::uint64_t producer_plan_generation) {
  if (state_ != QwenNumericalTapTransferState::kProducerPending ||
      producer_plan_generation != producer_plan_generation_) {
    return poison("Qwen inline tap producer identity is invalid");
  }
  state_ = QwenNumericalTapTransferState::kSnapshotReady;
  return submit_snapshot(driver);
}

Status QwenNumericalTapTransfer::submit_snapshot(TypedCopyDriver& driver) {
  if (state_ != QwenNumericalTapTransferState::kSnapshotReady) {
    return poison("Qwen tap snapshot copy is out of order");
  }
  const Status status = snapshot_copy_.submit(driver);
  if (!status.ok()) {
    state_ = QwenNumericalTapTransferState::kPoisoned;
    return status;
  }
  state_ = QwenNumericalTapTransferState::kSnapshotInFlight;
  return Status::Ok();
}

Status QwenNumericalTapTransfer::validate_copy_frontier(
    const CudaCompletionFrontier& frontier, const CudaTypedCopyPlan& plan,
    const char* message) {
  if (!frontier.publication_authorized() ||
      frontier.key().phase != CudaCompletionPhase::kCopy ||
      frontier.key().plan_generation != plan.plan_id() ||
      frontier.event_generation() != plan.completion_event_generation()) {
    return poison(message);
  }
  return Status::Ok();
}

Status QwenNumericalTapTransfer::complete_snapshot(
    const CudaCompletionFrontier& frontier) {
  if (state_ != QwenNumericalTapTransferState::kSnapshotInFlight) {
    return poison("Qwen tap snapshot completion is out of order");
  }
  const Status status = validate_copy_frontier(
      frontier, snapshot_copy_, "Qwen tap snapshot frontier is invalid");
  if (!status.ok()) return status;
  state_ = QwenNumericalTapTransferState::kHostReady;
  return Status::Ok();
}

Status QwenNumericalTapTransfer::submit_host(TypedCopyDriver& driver) {
  if (state_ != QwenNumericalTapTransferState::kHostReady) {
    return poison("Qwen tap host copy is out of order");
  }
  const Status status = host_copy_.submit(driver);
  if (!status.ok()) {
    state_ = QwenNumericalTapTransferState::kPoisoned;
    return status;
  }
  state_ = QwenNumericalTapTransferState::kHostInFlight;
  return Status::Ok();
}

Status QwenNumericalTapTransfer::complete_host(
    const CudaCompletionFrontier& frontier) {
  if (state_ != QwenNumericalTapTransferState::kHostInFlight) {
    return poison("Qwen tap host completion is out of order");
  }
  const Status status = validate_copy_frontier(
      frontier, host_copy_, "Qwen tap host frontier is invalid");
  if (!status.ok()) return status;
  state_ = QwenNumericalTapTransferState::kHostComplete;
  return Status::Ok();
}

Result<QwenNumericalTapReceipt> QwenNumericalTapTransfer::seal(
    std::span<const std::byte> observed_bytes) {
  if (state_ != QwenNumericalTapTransferState::kHostComplete ||
      observed_bytes.size() != bytes_) {
    return poison("Qwen tap writer bytes are incomplete");
  }
  auto digest = sha256(observed_bytes);
  if (!digest.ok()) {
    state_ = QwenNumericalTapTransferState::kPoisoned;
    return digest.status();
  }
  state_ = QwenNumericalTapTransferState::kSealed;
  return QwenNumericalTapReceipt{capture_index_, capture_generation_, bytes_,
                                 *digest};
}

Result<QwenNumericalTapInlineSnapshotDriver>
QwenNumericalTapInlineSnapshotDriver::Create(
    std::span<QwenNumericalTapTransfer> transfers,
    TypedCopyDriver& copy_driver,
    std::uint64_t producer_plan_generation) {
  if (transfers.empty() || producer_plan_generation == 0) {
    return Status::InvalidArgument(
        "Qwen inline tap snapshot identity is invalid");
  }
  for (std::size_t index = 0; index < transfers.size(); ++index) {
    if (transfers[index].capture_index() != index ||
        transfers[index].producer_plan_generation() !=
            producer_plan_generation ||
        transfers[index].state() !=
            QwenNumericalTapTransferState::kProducerPending) {
      return Status::FailedPrecondition(
          "Qwen inline tap transfers are not a complete ordered generation");
    }
  }
  return QwenNumericalTapInlineSnapshotDriver(
      transfers, copy_driver, producer_plan_generation);
}

Status QwenNumericalTapInlineSnapshotDriver::snapshot(
    const QwenBf16TapBinding& binding, DriverStreamHandle stream) {
  if (binding.capture_index >= transfers_.size() ||
      transfers_[binding.capture_index].capture_index() !=
          binding.capture_index ||
      stream == 0 ||
      transfers_[binding.capture_index].snapshot_stream() != stream) {
    return Status::FailedPrecondition(
        "Qwen inline tap binding identity is invalid");
  }
  return transfers_[binding.capture_index].submit_inline_snapshot(
      *copy_driver_, producer_plan_generation_);
}

}  // namespace pih
