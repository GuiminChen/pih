#include "pih/model/deepseek_production_rank_boundary_factory.h"

#include <chrono>

#include "pih/core/checked_math.h"

namespace pih {

Result<std::uint64_t> DeepSeekSteadyBoundaryClock::now_ns() {
  const auto elapsed = std::chrono::steady_clock::now().time_since_epoch();
  const auto ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
  if (ns <= 0) {
    return Status::Internal(
        "DeepSeek boundary steady clock returned invalid time");
  }
  return static_cast<std::uint64_t>(ns);
}

Result<std::unique_ptr<DeepSeekProductionRankBoundaryFactory>>
DeepSeekProductionRankBoundaryFactory::Create(
    DeepSeekRankBoundaryFactoryIdentity identity,
    DeepSeekBoundaryTransportDriver* incoming_transport,
    DeepSeekBoundaryTransportDriver* outgoing_transport,
    std::unique_ptr<DeepSeekRankBoundaryDeviceResources> resources,
    DeepSeekBoundaryPreparationClock& clock,
    DeepSeekBoundaryDriverOwnerFactory& driver_factory) {
  if (identity.engine_epoch == 0 || identity.world_size < 2 ||
      identity.world_size > 4 || identity.rank >= identity.world_size ||
      identity.context_identity == 0 || identity.boundary_stream == 0 ||
      identity.communicator_generation == 0 || identity.timeout_ns == 0 ||
      resources == nullptr ||
      (identity.rank == 0) != (incoming_transport == nullptr) ||
      (identity.rank + 1 == identity.world_size) !=
          (outgoing_transport == nullptr) ||
      resources->incoming_slot_count() !=
          (identity.rank == 0 ? 0U :
                                DeepSeekRankBoundaryDeviceResources::kCreditCount) ||
      resources->incoming_event_count() !=
          (identity.rank == 0 ? 0U :
                                DeepSeekRankBoundaryDeviceResources::kCreditCount) ||
      resources->outgoing_event_count() !=
          (identity.rank + 1 == identity.world_size
               ? 0U
               : DeepSeekRankBoundaryDeviceResources::kCreditCount)) {
    return Status::InvalidArgument(
        "DeepSeek production boundary factory identity is invalid");
  }
  if (identity.rank != 0 &&
      (identity.incoming_buffer_owner_ids[0] == 0 ||
       identity.incoming_buffer_owner_ids[1] == 0 ||
       identity.incoming_buffer_owner_ids[0] ==
           identity.incoming_buffer_owner_ids[1])) {
    return Status::InvalidArgument(
        "DeepSeek incoming boundary owner identities are invalid");
  }
  auto sequencer = DeepSeekNcclOperationSequencer::Create(1);
  auto incoming_tracker = DeepSeekBoundaryCreditTracker::Create(2);
  auto outgoing_tracker = DeepSeekBoundaryCreditTracker::Create(2);
  if (!sequencer.ok()) return sequencer.status();
  if (!incoming_tracker.ok()) return incoming_tracker.status();
  if (!outgoing_tracker.ok()) return outgoing_tracker.status();
  return std::unique_ptr<DeepSeekProductionRankBoundaryFactory>(
      new DeepSeekProductionRankBoundaryFactory(
          identity, incoming_transport, outgoing_transport,
          std::move(resources), clock, driver_factory,
          std::move(*sequencer), std::move(*incoming_tracker),
          std::move(*outgoing_tracker)));
}

Result<DeepSeekPreparedRankBoundaries>
DeepSeekProductionRankBoundaryFactory::prepare(
    DeepSeekPipelineTransaction& transaction,
    const DeepSeekStagePlan& stage, std::uint32_t world_size,
    std::uint32_t wire_token_count,
    const std::optional<DeepSeekBoundarySendSource>& outgoing_source) {
  if (transaction.state() != DeepSeekPipelineTransactionState::kPrepared ||
      transaction.descriptor().engine_epoch != identity_.engine_epoch ||
      stage.rank != identity_.rank || world_size != identity_.world_size ||
      wire_token_count == 0 ||
      stage.owns_embedding != (identity_.rank == 0) ||
      stage.owns_lm_head != (identity_.rank + 1 == identity_.world_size) ||
      outgoing_source.has_value() != (outgoing_transport_ != nullptr)) {
    return Status::InvalidArgument(
        "DeepSeek boundary preparation differs from factory topology");
  }
  auto submit_ns = clock_->now_ns();
  if (!submit_ns.ok()) return submit_ns.status();
  auto deadline_ns = checked_add_u64(*submit_ns, identity_.timeout_ns);
  if (!deadline_ns.ok()) return deadline_ns.status();

  DeepSeekPreparedRankBoundaries result;
  result.drivers.clock = clock_;
  if (incoming_transport_ != nullptr) {
    auto owner = driver_factory_->create(*incoming_transport_);
    if (!owner.ok()) return owner.status();
    auto incoming = DeepSeekBoundaryOperationBuilder::CreateRecv(
        transaction, world_size, identity_.rank - 1,
        identity_.communicator_generation, wire_token_count, *resources_,
        identity_.incoming_buffer_owner_ids, identity_.context_identity,
        identity_.boundary_stream, sequencer_, incoming_tracker_,
        *submit_ns, *deadline_ns);
    if (!incoming.ok()) return incoming.status();
    result.incoming = std::move(*incoming);
    result.drivers.incoming.owner = std::move(*owner);
  }
  if (outgoing_transport_ != nullptr) {
    auto owner = driver_factory_->create(*outgoing_transport_);
    if (!owner.ok()) return owner.status();
    auto outgoing = DeepSeekBoundaryOperationBuilder::CreateSend(
        transaction, world_size, identity_.rank,
        identity_.communicator_generation, wire_token_count,
        *outgoing_source, *resources_, identity_.boundary_stream,
        sequencer_, outgoing_tracker_, *submit_ns, *deadline_ns);
    if (!outgoing.ok()) return outgoing.status();
    result.outgoing = std::move(*outgoing);
    result.drivers.outgoing.owner = std::move(*owner);
  }
  return result;
}

}  // namespace pih
