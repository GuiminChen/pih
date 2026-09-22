#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "pih/backend/cuda/typed_copy_plan.h"
#include "pih/model/qwen3_kv_semantic_observation_plan.h"

namespace pih {

struct QwenSemanticObservationTransferIdentity final {
  std::uint64_t first_plan_id;
  std::uintptr_t context_identity;
  DriverStreamHandle diagnostic_stream;
  std::uint64_t completion_event_generation;
};

enum class QwenSemanticObservationTransferState : std::uint8_t {
  kPrepared,
  kSubmitted,
  kPoisoned,
};

class QwenSemanticObservationTransfer final {
 public:
  static constexpr std::uint64_t kFinalLogitsBytes = 151936ULL * 4;

  static Result<QwenSemanticObservationTransfer> Create(
      CudaCopyEndpoint logits_source, CudaCopyEndpoint logits_destination,
      CudaCopyEndpoint kv_source, CudaCopyEndpoint kv_destination,
      const QwenKvSemanticObservationPlan& kv_plan,
      QwenSemanticObservationTransferIdentity identity);

  Status submit(TypedCopyDriver& driver);

  [[nodiscard]] QwenSemanticObservationTransferState state() const noexcept {
    return state_;
  }
  [[nodiscard]] std::span<const CudaTypedCopyPlan> plans() const noexcept {
    return plans_;
  }
  [[nodiscard]] std::uint64_t logits_bytes() const noexcept {
    return kFinalLogitsBytes;
  }
  [[nodiscard]] std::uint64_t kv_bytes() const noexcept { return kv_bytes_; }
  [[nodiscard]] std::uint64_t completion_event_generation() const noexcept {
    return completion_event_generation_;
  }
  [[nodiscard]] DriverStreamHandle diagnostic_stream() const noexcept {
    return diagnostic_stream_;
  }
  [[nodiscard]] std::uintptr_t context_identity() const noexcept {
    return context_identity_;
  }

 private:
  QwenSemanticObservationTransfer(
      std::vector<CudaTypedCopyPlan> plans, std::uint64_t kv_bytes,
      std::uint64_t completion_event_generation,
      DriverStreamHandle diagnostic_stream, std::uintptr_t context_identity)
      : plans_(std::move(plans)), kv_bytes_(kv_bytes),
        completion_event_generation_(completion_event_generation),
        diagnostic_stream_(diagnostic_stream),
        context_identity_(context_identity) {}

  std::vector<CudaTypedCopyPlan> plans_;
  std::uint64_t kv_bytes_;
  std::uint64_t completion_event_generation_;
  DriverStreamHandle diagnostic_stream_;
  std::uintptr_t context_identity_;
  QwenSemanticObservationTransferState state_ =
      QwenSemanticObservationTransferState::kPrepared;
};

}  // namespace pih
