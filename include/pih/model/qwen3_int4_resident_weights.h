#pragma once

#include <cstdint>

#include "pih/backend/cuda/completion_event_slot.h"
#include "pih/core/buffer.h"
#include "pih/model/qwen3_int4_weight_resource_set.h"
#include "pih/model/qwen3_int4_weight_upload_plan.h"

namespace pih {

enum class QwenInt4ResidentWeightState : std::uint8_t {
  kPrepared,
  kSubmitted,
  kPublished,
  kPoisoned,
};

class QwenInt4ResidentWeights final {
 public:
  static Result<QwenInt4ResidentWeights> Create(
      const QwenInt4ArtifactLayout& layout,
      const QwenInt4LinearShapeLedger& ledger, std::uint64_t engine_epoch,
      Allocator& allocator,
      CudaCopyEndpoint canonical_file, std::int32_t owning_rank,
      std::uint64_t destination_owner_id,
      std::uintptr_t primary_context_identity, DriverStreamHandle stream,
      DriverEventHandle completion_event,
      std::uint64_t completion_event_generation,
      std::uint64_t first_plan_id, std::uint64_t submit_ns,
      std::uint64_t deadline_ns);

  QwenInt4ResidentWeights(const QwenInt4ResidentWeights&) = delete;
  QwenInt4ResidentWeights& operator=(const QwenInt4ResidentWeights&) = delete;
  QwenInt4ResidentWeights(QwenInt4ResidentWeights&&) noexcept = default;
  QwenInt4ResidentWeights& operator=(QwenInt4ResidentWeights&&) noexcept = default;

  Status submit(TypedCopyDriver& copy_driver,
                CompletionEventDriver& event_driver);
  Status poll(CompletionEventDriver& event_driver,
              CompletionEvidenceProvider& evidence_provider);
  Status expire(std::uint64_t now_ns);
  Result<const QwenInt4WeightResourceSet*> resources() const;
  [[nodiscard]] QwenInt4ResidentWeightState state() const noexcept {
    return state_;
  }

 private:
  QwenInt4ResidentWeights(Buffer backing,
                          QwenInt4WeightResourceSet pending_resources,
                          QwenInt4WeightUploadPlan upload,
                          CompletionEventSlot event_slot,
                          CudaCompletionFrontier frontier,
                          DriverStreamHandle stream,
                          std::uint64_t event_generation)
      : backing_(std::move(backing)),
        pending_resources_(std::move(pending_resources)),
        upload_(std::move(upload)), event_slot_(std::move(event_slot)),
        frontier_(std::move(frontier)), stream_(stream),
        event_generation_(event_generation) {}

  Buffer backing_;
  QwenInt4WeightResourceSet pending_resources_;
  QwenInt4WeightUploadPlan upload_;
  CompletionEventSlot event_slot_;
  CudaCompletionFrontier frontier_;
  DriverStreamHandle stream_;
  std::uint64_t event_generation_;
  QwenInt4ResidentWeightState state_ =
      QwenInt4ResidentWeightState::kPrepared;
};

}  // namespace pih
