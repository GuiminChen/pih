#pragma once

#include <cstdint>
#include <optional>

#include "pih/backend/cuda/completion_event_slot.h"
#include "pih/model/qwen3_bf16_tap_evidence_run.h"

namespace pih {

enum class QwenBf16TapEventPipelineState : std::uint8_t {
  kCapturing = 0,
  kSnapshotRecorded,
  kHostReady,
  kHostRecorded,
  kHostComplete,
  kSealed,
  kPoisoned,
};

class QwenBf16TapEventPipeline final
    : public QwenBf16TapSnapshotFrontierRecorder {
 public:
  static Result<QwenBf16TapEventPipeline> Create(
      QwenBf16TapEvidenceRun evidence_run,
      DriverEventHandle event,
      std::uint64_t epoch);

  Result<QwenNumericalTapInlineSnapshotDriver> inline_snapshot_driver(
      TypedCopyDriver& copy_driver);
  Status record_snapshot(CompletionEventDriver& event_driver,
                         std::uint64_t submit_ns,
                         std::uint64_t deadline_ns) override;
  Status poll_snapshot(CompletionEventDriver& event_driver,
                       CompletionEvidenceProvider& evidence_provider);
  Status submit_host_and_record(TypedCopyDriver& copy_driver,
                                CompletionEventDriver& event_driver,
                                std::uint64_t submit_ns,
                                std::uint64_t deadline_ns);
  Status poll_host(CompletionEventDriver& event_driver,
                   CompletionEvidenceProvider& evidence_provider);
  Status expire(std::uint64_t now_ns);
  Result<QwenNumericalTapRunReceipt> seal(
      const QwenNumericalTapArenas& arenas);

  [[nodiscard]] QwenBf16TapEventPipelineState state() const noexcept {
    return state_;
  }
  [[nodiscard]] std::uint64_t host_event_generation() const noexcept {
    return evidence_run_.identity().host_event_generation;
  }
  [[nodiscard]] DriverStreamHandle diagnostic_stream() const noexcept {
    return evidence_run_.identity().diagnostic_stream;
  }

 private:
  QwenBf16TapEventPipeline(QwenBf16TapEvidenceRun evidence_run,
                           CompletionEventSlot event_slot,
                           std::uint64_t epoch)
      : evidence_run_(std::move(evidence_run)),
        event_slot_(std::move(event_slot)), epoch_(epoch) {}
  Status poison(const char* message);
  Result<CudaCompletionFrontier> make_frontier(
      std::uint64_t plan_id, std::uint64_t event_generation,
      std::uint64_t submit_ns, std::uint64_t deadline_ns) const;

  QwenBf16TapEvidenceRun evidence_run_;
  CompletionEventSlot event_slot_;
  std::uint64_t epoch_;
  std::optional<CudaCompletionFrontier> frontier_;
  QwenBf16TapEventPipelineState state_ =
      QwenBf16TapEventPipelineState::kCapturing;
};

}  // namespace pih
