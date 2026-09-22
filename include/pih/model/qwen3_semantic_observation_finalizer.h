#pragma once

#include <cstdint>

#include "pih/model/qwen3_bf16_synchronous_backend.h"
#include "pih/model/qwen3_semantic_observation_pipeline.h"
#include "pih/model/qwen3_semantic_pinned_arenas.h"

namespace pih {

struct QwenSemanticObservationSources final {
  CudaCopyEndpoint logits;
  CudaCopyEndpoint kv;
};

struct QwenSemanticObservationFinalizerIdentity final {
  std::uint64_t epoch;
  std::uint64_t first_plan_id;
  std::uint64_t frontier_plan_id;
  std::uint64_t completion_event_generation;
  std::uint64_t timeout_ns;
  std::uintptr_t context_identity;
  DriverStreamHandle diagnostic_stream;
  DriverEventHandle diagnostic_event;
  std::uint32_t rank;
  std::int32_t numa_node;
  std::uint64_t pinned_logits_owner_id;
  std::uint64_t pinned_kv_owner_id;
};

struct QwenSemanticObservationFinalizerDrivers final {
  RegisteredPinnedAllocator* pinned_allocator;
  PinnedPlacementVerifier* placement;
  TypedCopyDriver* copy;
  CompletionEventDriver* events;
  CompletionEvidenceProvider* evidence;
  QwenBf16MonotonicClock* clock;
  QwenBf16PollWaiter* waiter;
};

class QwenSemanticObservationFinalizer final {
 public:
  static Result<QwenSemanticOutcome> Run(
      const QwenKvSemanticObservationPlan& kv_plan,
      QwenSemanticObservationSources sources,
      QwenSemanticObservationFinalizerIdentity identity,
      QwenSemanticObservationFinalizerDrivers drivers,
      std::uint64_t device_peak_bytes, std::uint64_t pinned_peak_bytes,
      QwenSemanticOutcomeRecorder recorder);
};

}  // namespace pih
