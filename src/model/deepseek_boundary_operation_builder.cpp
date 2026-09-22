#include "pih/model/deepseek_boundary_operation_builder.h"

namespace pih {
namespace {

Result<std::unique_ptr<DeepSeekTrackedBoundaryOperation>> finish_reserved(
    DeepSeekPipelineTransaction& transaction,
    DeepSeekNcclP2pManifest manifest, Buffer& buffer,
    std::uint64_t buffer_owner_id, std::uintptr_t context_identity,
    DriverStreamHandle boundary_stream, DriverEventHandle completion_event,
    DeepSeekNcclOperationSequencer& sequencer,
    DeepSeekBoundaryCreditTracker& tracker,
    DeepSeekBoundaryCreditHandle credit,
    std::uint64_t submit_ns, std::uint64_t deadline_ns) {
  auto prepared = DeepSeekPreparedBoundaryOperation::Create(
      transaction, manifest, buffer, buffer_owner_id, context_identity,
      boundary_stream, completion_event, sequencer, submit_ns, deadline_ns);
  if (!prepared.ok()) {
    const auto rollback = tracker.abort_prepare(credit);
    return rollback.ok() ? prepared.status() : rollback;
  }
  auto tracked = DeepSeekTrackedBoundaryOperation::Create(
      std::make_unique<DeepSeekPreparedBoundaryOperation>(
          std::move(*prepared)), tracker, credit);
  if (!tracked.ok()) {
    const auto rollback = tracker.abort_prepare(credit);
    return rollback.ok() ? tracked.status() : rollback;
  }
  return std::move(*tracked);
}

}  // namespace

Result<std::unique_ptr<DeepSeekTrackedBoundaryOperation>>
DeepSeekBoundaryOperationBuilder::CreateSend(
    DeepSeekPipelineTransaction& transaction, std::uint32_t world_size,
    std::uint32_t directed_boundary_id,
    std::uint64_t communicator_generation, std::uint32_t wire_token_count,
    const DeepSeekBoundarySendSource& source,
    DeepSeekRankBoundaryDeviceResources& resources,
    DriverStreamHandle boundary_stream,
    DeepSeekNcclOperationSequencer& sequencer,
    DeepSeekBoundaryCreditTracker& tracker,
    std::uint64_t submit_ns, std::uint64_t deadline_ns) {
  auto manifest = DeepSeekBoundaryManifestBuilder::CreateSend(
      transaction.descriptor(), world_size, directed_boundary_id,
      communicator_generation, wire_token_count, source);
  if (!manifest.ok()) return manifest.status();
  auto credit = tracker.reserve(manifest->pipeline_plan_sequence,
                                manifest->operation_ordinal);
  if (!credit.ok()) return credit.status();
  const auto completion_event = resources.outgoing_event(credit->credit_index);
  if (completion_event == 0) {
    const auto rollback = tracker.abort_prepare(*credit);
    return rollback.ok()
               ? Status::FailedPrecondition(
                     "DeepSeek send credit has no completion event")
               : rollback;
  }
  return finish_reserved(transaction, *manifest, source.buffer(),
                source.buffer_owner_id(), source.context_identity(),
                boundary_stream, completion_event, sequencer, tracker,
                *credit, submit_ns, deadline_ns);
}

Result<std::unique_ptr<DeepSeekTrackedBoundaryOperation>>
DeepSeekBoundaryOperationBuilder::CreateRecv(
    DeepSeekPipelineTransaction& transaction, std::uint32_t world_size,
    std::uint32_t directed_boundary_id,
    std::uint64_t communicator_generation, std::uint32_t wire_token_count,
    DeepSeekRankBoundaryDeviceResources& resources,
    const std::array<std::uint64_t,
                     DeepSeekRankBoundaryDeviceResources::kCreditCount>&
        buffer_owner_ids,
    std::uintptr_t context_identity, DriverStreamHandle boundary_stream,
    DeepSeekNcclOperationSequencer& sequencer,
    DeepSeekBoundaryCreditTracker& tracker,
    std::uint64_t submit_ns, std::uint64_t deadline_ns) {
  auto ordinal = DeepSeekNcclOrdinalPlan::Create(
      world_size, transaction.descriptor().plan_sequence,
      directed_boundary_id, DeepSeekNcclRole::kRecv);
  if (!ordinal.ok()) return ordinal.status();
  auto credit = tracker.reserve(transaction.descriptor().plan_sequence,
                                ordinal->rank_local_ordinal);
  if (!credit.ok()) return credit.status();
  Buffer* credit_slot = resources.incoming_slot(credit->credit_index);
  const auto completion_event = resources.incoming_event(credit->credit_index);
  const auto buffer_owner_id = buffer_owner_ids[credit->credit_index];
  if (credit_slot == nullptr || completion_event == 0 || buffer_owner_id == 0) {
    const auto rollback = tracker.abort_prepare(*credit);
    return rollback.ok()
               ? Status::FailedPrecondition(
                     "DeepSeek receive credit has no device resource")
               : rollback;
  }
  auto manifest = DeepSeekBoundaryManifestBuilder::CreateRecv(
      transaction.descriptor(), world_size, directed_boundary_id,
      communicator_generation, wire_token_count, *credit_slot,
      buffer_owner_id, context_identity);
  if (!manifest.ok()) {
    const auto rollback = tracker.abort_prepare(*credit);
    return rollback.ok() ? manifest.status() : rollback;
  }
  return finish_reserved(transaction, *manifest, *credit_slot, buffer_owner_id,
                context_identity, boundary_stream, completion_event,
                sequencer, tracker, *credit, submit_ns, deadline_ns);
}

}  // namespace pih
