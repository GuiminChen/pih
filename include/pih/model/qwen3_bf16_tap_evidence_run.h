#pragma once

#include <cstdint>

#include "pih/model/qwen3_bf16_tap_transfer_plan.h"
#include "pih/model/qwen3_numerical_tap_run.h"

namespace pih {

enum class QwenBf16TapEvidenceRunState : std::uint8_t {
  kPrepared = 0,
  kCapturing,
  kHostReady,
  kHostInFlight,
  kHostComplete,
  kSealed,
  kPoisoned,
};

class QwenBf16TapEvidenceRun final {
 public:
  static Result<QwenBf16TapEvidenceRun> Create(
      const QwenNumericalTapPlan& taps,
      QwenBf16TapTransferPlan transfer_plan);

  Result<QwenNumericalTapInlineSnapshotDriver> inline_snapshot_driver(
      TypedCopyDriver& copy_driver);
  Status complete_snapshot_batch(const CudaCompletionFrontier& frontier);
  Status submit_host_batch(TypedCopyDriver& copy_driver);
  Status complete_host_batch(const CudaCompletionFrontier& frontier);
  Result<QwenNumericalTapRunReceipt> seal(
      const QwenNumericalTapArenas& arenas);

  [[nodiscard]] QwenBf16TapEvidenceRunState state() const noexcept {
    return state_;
  }
  [[nodiscard]] const QwenBf16TapTransferIdentity& identity() const noexcept {
    return transfer_plan_.identity();
  }

 private:
  QwenBf16TapEvidenceRun(QwenBf16TapTransferPlan transfer_plan,
                         QwenNumericalTapRun run)
      : transfer_plan_(std::move(transfer_plan)), run_(std::move(run)) {}
  Status poison(const char* message);

  QwenBf16TapTransferPlan transfer_plan_;
  QwenNumericalTapRun run_;
  QwenBf16TapEvidenceRunState state_ =
      QwenBf16TapEvidenceRunState::kPrepared;
};

}  // namespace pih
