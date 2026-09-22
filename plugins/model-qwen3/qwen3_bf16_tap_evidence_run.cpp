#include "pih/model/qwen3_bf16_tap_evidence_run.h"

namespace pih {

Result<QwenBf16TapEvidenceRun> QwenBf16TapEvidenceRun::Create(
    const QwenNumericalTapPlan& taps,
    QwenBf16TapTransferPlan transfer_plan) {
  if (transfer_plan.size() != taps.captures().size()) {
    return Status::InvalidArgument(
        "Qwen tap evidence run transfer count is invalid");
  }
  auto run = QwenNumericalTapRun::Create(
      taps, transfer_plan.identity().capture_generation);
  if (!run.ok()) return run.status();
  return QwenBf16TapEvidenceRun(std::move(transfer_plan), std::move(*run));
}

Status QwenBf16TapEvidenceRun::poison(const char* message) {
  state_ = QwenBf16TapEvidenceRunState::kPoisoned;
  return Status::FailedPrecondition(message);
}

Result<QwenNumericalTapInlineSnapshotDriver>
QwenBf16TapEvidenceRun::inline_snapshot_driver(
    TypedCopyDriver& copy_driver) {
  if (state_ != QwenBf16TapEvidenceRunState::kPrepared) {
    return poison("Qwen tap evidence run is not prepared");
  }
  auto driver = QwenNumericalTapInlineSnapshotDriver::Create(
      transfer_plan_.transfers(), copy_driver,
      transfer_plan_.identity().producer_plan_generation);
  if (!driver.ok()) {
    state_ = QwenBf16TapEvidenceRunState::kPoisoned;
    return driver.status();
  }
  state_ = QwenBf16TapEvidenceRunState::kCapturing;
  return driver;
}

Status QwenBf16TapEvidenceRun::complete_snapshot_batch(
    const CudaCompletionFrontier& frontier) {
  if (state_ != QwenBf16TapEvidenceRunState::kCapturing) {
    return poison("Qwen tap snapshot batch is out of order");
  }
  for (const auto& transfer : transfer_plan_.transfers()) {
    if (transfer.state() !=
        QwenNumericalTapTransferState::kSnapshotInFlight) {
      return poison("Qwen tap snapshot batch is incomplete");
    }
  }
  for (auto& transfer : transfer_plan_.transfers()) {
    const Status status = transfer.complete_snapshot(frontier);
    if (!status.ok()) {
      state_ = QwenBf16TapEvidenceRunState::kPoisoned;
      return status;
    }
  }
  state_ = QwenBf16TapEvidenceRunState::kHostReady;
  return Status::Ok();
}

Status QwenBf16TapEvidenceRun::submit_host_batch(
    TypedCopyDriver& copy_driver) {
  if (state_ != QwenBf16TapEvidenceRunState::kHostReady) {
    return poison("Qwen tap host batch is out of order");
  }
  for (const auto& transfer : transfer_plan_.transfers()) {
    if (transfer.state() != QwenNumericalTapTransferState::kHostReady) {
      return poison("Qwen tap host batch is incomplete");
    }
  }
  for (auto& transfer : transfer_plan_.transfers()) {
    const Status status = transfer.submit_host(copy_driver);
    if (!status.ok()) {
      state_ = QwenBf16TapEvidenceRunState::kPoisoned;
      return status;
    }
  }
  state_ = QwenBf16TapEvidenceRunState::kHostInFlight;
  return Status::Ok();
}

Status QwenBf16TapEvidenceRun::complete_host_batch(
    const CudaCompletionFrontier& frontier) {
  if (state_ != QwenBf16TapEvidenceRunState::kHostInFlight) {
    return poison("Qwen tap host completion batch is out of order");
  }
  for (const auto& transfer : transfer_plan_.transfers()) {
    if (transfer.state() != QwenNumericalTapTransferState::kHostInFlight) {
      return poison("Qwen tap host completion batch is incomplete");
    }
  }
  for (auto& transfer : transfer_plan_.transfers()) {
    const Status status = transfer.complete_host(frontier);
    if (!status.ok()) {
      state_ = QwenBf16TapEvidenceRunState::kPoisoned;
      return status;
    }
  }
  state_ = QwenBf16TapEvidenceRunState::kHostComplete;
  return Status::Ok();
}

Result<QwenNumericalTapRunReceipt> QwenBf16TapEvidenceRun::seal(
    const QwenNumericalTapArenas& arenas) {
  if (state_ != QwenBf16TapEvidenceRunState::kHostComplete ||
      arenas.arena_bytes() == 0) {
    return poison("Qwen tap evidence run cannot be sealed");
  }
  for (std::size_t index = 0; index < transfer_plan_.size(); ++index) {
    auto bytes = arenas.pinned_capture(index);
    if (!bytes.ok()) {
      state_ = QwenBf16TapEvidenceRunState::kPoisoned;
      return bytes.status();
    }
    auto receipt = transfer_plan_.transfers()[index].seal(*bytes);
    if (!receipt.ok()) {
      state_ = QwenBf16TapEvidenceRunState::kPoisoned;
      return receipt.status();
    }
    const Status recorded = run_.record(*receipt);
    if (!recorded.ok()) {
      state_ = QwenBf16TapEvidenceRunState::kPoisoned;
      return recorded;
    }
  }
  auto receipt = run_.seal();
  if (!receipt.ok()) {
    state_ = QwenBf16TapEvidenceRunState::kPoisoned;
    return receipt.status();
  }
  state_ = QwenBf16TapEvidenceRunState::kSealed;
  return receipt;
}

}  // namespace pih
