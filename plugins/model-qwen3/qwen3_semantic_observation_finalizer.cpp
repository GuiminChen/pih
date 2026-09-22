#include "pih/model/qwen3_semantic_observation_finalizer.h"

#include "pih/core/checked_math.h"

namespace pih {

Result<QwenSemanticOutcome> QwenSemanticObservationFinalizer::Run(
    const QwenKvSemanticObservationPlan& kv_plan,
    QwenSemanticObservationSources sources,
    QwenSemanticObservationFinalizerIdentity identity,
    QwenSemanticObservationFinalizerDrivers drivers,
    std::uint64_t device_peak_bytes, std::uint64_t pinned_peak_bytes,
    QwenSemanticOutcomeRecorder recorder) {
  if (identity.epoch == 0 || identity.first_plan_id == 0 ||
      identity.frontier_plan_id == 0 ||
      identity.completion_event_generation == 0 || identity.timeout_ns == 0 ||
      identity.context_identity == 0 || identity.diagnostic_stream == 0 ||
      identity.diagnostic_event == 0 || identity.rank == UINT32_MAX ||
      identity.numa_node < 0 || identity.pinned_logits_owner_id == 0 ||
      identity.pinned_kv_owner_id == 0 || device_peak_bytes == 0 ||
      pinned_peak_bytes == 0 || drivers.pinned_allocator == nullptr ||
      drivers.placement == nullptr || drivers.copy == nullptr ||
      drivers.events == nullptr || drivers.evidence == nullptr ||
      drivers.clock == nullptr || drivers.waiter == nullptr) {
    return Status::InvalidArgument(
        "Qwen semantic observation finalizer inputs are invalid");
  }
  auto arenas = QwenSemanticPinnedArenas::AllocateVerified(
      kv_plan, *drivers.pinned_allocator, *drivers.placement,
      static_cast<std::int32_t>(identity.rank), identity.numa_node);
  if (!arenas.ok()) return arenas.status();
  auto logits_destination = arenas->logits_endpoint(
      identity.pinned_logits_owner_id, identity.rank);
  if (!logits_destination.ok()) return logits_destination.status();
  auto kv_destination =
      arenas->kv_endpoint(identity.pinned_kv_owner_id, identity.rank);
  if (!kv_destination.ok()) return kv_destination.status();
  auto transfer = QwenSemanticObservationTransfer::Create(
      sources.logits, *logits_destination, sources.kv, *kv_destination,
      kv_plan,
      {identity.first_plan_id, identity.context_identity,
       identity.diagnostic_stream, identity.completion_event_generation});
  if (!transfer.ok()) return transfer.status();
  auto pipeline = QwenSemanticObservationPipeline::Create(
      std::move(*transfer), identity.diagnostic_event, identity.epoch,
      identity.rank, identity.frontier_plan_id);
  if (!pipeline.ok()) return pipeline.status();
  auto submit_ns = drivers.clock->now_ns();
  if (!submit_ns.ok()) return submit_ns.status();
  auto deadline_ns = checked_add_u64(*submit_ns, identity.timeout_ns);
  if (!deadline_ns.ok()) return deadline_ns.status();
  Status status = pipeline->submit(*drivers.copy, *drivers.events, *submit_ns,
                                   *deadline_ns);
  if (status.ok()) {
    status = pipeline->await(*drivers.events, *drivers.evidence,
                             *drivers.clock, *drivers.waiter);
  }
  if (status.ok()) {
    status = pipeline->publish(arenas->logits(), arenas->kv(), recorder);
  }
  if (status.ok()) {
    status = recorder.observe_capacity(device_peak_bytes, pinned_peak_bytes);
  }
  if (!status.ok()) return status;
  return recorder.seal();
}

}  // namespace pih
