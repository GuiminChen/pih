#pragma once

#include "pih/model/qwen3_bf16_tap_event_pipeline.h"
#include "pih/model/qwen3_bf16_tap_source_plan.h"

namespace pih {

class QwenBf16TapFixturePreparation final {
 public:
  static Result<QwenBf16TapFixturePreparation> Create(
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
      std::uint64_t epoch);

  QwenBf16TapFixturePreparation(
      const QwenBf16TapFixturePreparation&) = delete;
  QwenBf16TapFixturePreparation& operator=(
      const QwenBf16TapFixturePreparation&) = delete;
  QwenBf16TapFixturePreparation(
      QwenBf16TapFixturePreparation&&) noexcept = default;
  QwenBf16TapFixturePreparation& operator=(
      QwenBf16TapFixturePreparation&&) noexcept = default;

  [[nodiscard]] const QwenBf16TapBindingPlan& bindings() const noexcept {
    return bindings_;
  }
  [[nodiscard]] QwenBf16TapEventPipeline& pipeline() noexcept {
    return pipeline_;
  }
  [[nodiscard]] const QwenNumericalTapArenas& arenas() const noexcept {
    return arenas_;
  }

 private:
  QwenBf16TapFixturePreparation(QwenBf16TapBindingPlan bindings,
                                QwenNumericalTapArenas arenas,
                                QwenBf16TapEventPipeline pipeline)
      : bindings_(std::move(bindings)), arenas_(std::move(arenas)),
        pipeline_(std::move(pipeline)) {}

  QwenBf16TapBindingPlan bindings_;
  QwenNumericalTapArenas arenas_;
  QwenBf16TapEventPipeline pipeline_;
};

}  // namespace pih
