#include "pih/model/qwen3_bf16_tap_fixture_preparation.h"

namespace pih {

Result<QwenBf16TapFixturePreparation>
QwenBf16TapFixturePreparation::Create(
    const QwenNumericalTapPlan& taps,
    const QwenBf16CommandBuffer& commands,
    const QwenBf16ResourceSet& resources,
    const QwenKvBlockTable& block_table,
    std::span<const QwenKvSlotState> projected_slot_states,
    std::uint64_t activation_first_position,
    std::uint64_t source_owner_id_base,
    Allocator& device_allocator,
    RegisteredPinnedAllocator& pinned_allocator,
    PinnedPlacementVerifier& placement_verifier,
    std::int32_t numa_node,
    QwenBf16TapTransferIdentity transfer_identity,
    DriverEventHandle diagnostic_event,
    std::uint64_t epoch) {
  if (transfer_identity.producer_plan_generation !=
          resources.request_generation() ||
      transfer_identity.rank !=
          static_cast<std::uint32_t>(resources.owning_rank())) {
    return Status::InvalidArgument(
        "Qwen tap fixture producer identity differs from step resources");
  }
  auto bindings = QwenBf16TapBindingPlan::Create(taps, commands);
  if (!bindings.ok()) return bindings.status();
  auto arenas = QwenNumericalTapArenas::AllocateVerified(
      taps, device_allocator, pinned_allocator, placement_verifier,
      resources.owning_rank(), numa_node);
  if (!arenas.ok()) return arenas.status();
  auto sources = QwenBf16TapSourcePlan::Create(
      taps, *bindings, resources, block_table, projected_slot_states,
      activation_first_position, source_owner_id_base);
  if (!sources.ok()) return sources.status();
  auto transfers = QwenBf16TapTransferPlan::Create(
      taps, sources->sources(), *arenas, transfer_identity);
  if (!transfers.ok()) return transfers.status();
  auto evidence = QwenBf16TapEvidenceRun::Create(
      taps, std::move(*transfers));
  if (!evidence.ok()) return evidence.status();
  auto pipeline = QwenBf16TapEventPipeline::Create(
      std::move(*evidence), diagnostic_event, epoch);
  if (!pipeline.ok()) return pipeline.status();
  return QwenBf16TapFixturePreparation(
      std::move(*bindings), std::move(*arenas), std::move(*pipeline));
}

}  // namespace pih
