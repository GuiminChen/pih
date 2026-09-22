#pragma once

#include <cstdint>
#include <optional>
#include <span>

#include "pih/backend/cuda/completion_event_slot.h"
#include "pih/model/qwen3_semantic_observation_transfer.h"
#include "pih/model/qwen3_semantic_outcome_recorder.h"
#include "pih/model/qwen3_bf16_synchronous_backend.h"

namespace pih {

enum class QwenSemanticObservationPipelineState : std::uint8_t {
  kPrepared,
  kRecorded,
  kComplete,
  kPublished,
  kPoisoned,
};

class QwenSemanticObservationPipeline final {
 public:
  static Result<QwenSemanticObservationPipeline> Create(
      QwenSemanticObservationTransfer transfer, DriverEventHandle event,
      std::uint64_t epoch, std::uint32_t rank,
      std::uint64_t frontier_plan_id);

  Status submit(TypedCopyDriver& copy_driver,
                CompletionEventDriver& event_driver,
                std::uint64_t submit_ns, std::uint64_t deadline_ns);
  Status poll(CompletionEventDriver& event_driver,
              CompletionEvidenceProvider& evidence_provider);
  Status await(CompletionEventDriver& event_driver,
               CompletionEvidenceProvider& evidence_provider,
               QwenBf16MonotonicClock& clock,
               QwenBf16PollWaiter& waiter);
  Status publish(std::span<const std::byte> logits,
                 std::span<const std::byte> kv,
                 QwenSemanticOutcomeRecorder& recorder);

  [[nodiscard]] QwenSemanticObservationPipelineState state() const noexcept {
    return state_;
  }

 private:
  QwenSemanticObservationPipeline(
      QwenSemanticObservationTransfer transfer, CompletionEventSlot event_slot,
      std::uint64_t epoch, std::uint32_t rank,
      std::uint64_t frontier_plan_id)
      : transfer_(std::move(transfer)), event_slot_(std::move(event_slot)),
        epoch_(epoch), rank_(rank), frontier_plan_id_(frontier_plan_id) {}
  Status poison(const char* message);

  QwenSemanticObservationTransfer transfer_;
  CompletionEventSlot event_slot_;
  std::uint64_t epoch_;
  std::uint32_t rank_;
  std::uint64_t frontier_plan_id_;
  std::optional<CudaCompletionFrontier> frontier_;
  QwenSemanticObservationPipelineState state_ =
      QwenSemanticObservationPipelineState::kPrepared;
};

}  // namespace pih
