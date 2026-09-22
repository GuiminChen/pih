#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "pih/model/qwen3_bf16_tap_source_plan.h"
#include "pih/model/qwen3_numerical_tap_arenas.h"
#include "pih/model/qwen3_numerical_tap_transfer.h"

namespace pih {

struct QwenBf16TapTransferIdentity final {
  std::uint64_t capture_generation;
  std::uint64_t producer_plan_generation;
  std::uint64_t snapshot_copy_plan_id;
  std::uint64_t host_copy_plan_id;
  std::uint64_t snapshot_event_generation;
  std::uint64_t host_event_generation;
  std::uintptr_t primary_context_identity;
  DriverStreamHandle execution_stream;
  DriverStreamHandle diagnostic_stream;
  std::uint64_t device_arena_owner_id;
  std::uint64_t pinned_arena_owner_id;
  std::uint32_t rank;
};

class QwenBf16TapTransferPlan final {
 public:
  static Result<QwenBf16TapTransferPlan> Create(
      const QwenNumericalTapPlan& taps,
      std::span<const CudaCopyEndpoint> sources,
      const QwenNumericalTapArenas& arenas,
      QwenBf16TapTransferIdentity identity);

  [[nodiscard]] std::size_t size() const noexcept { return transfers_.size(); }
  [[nodiscard]] std::span<QwenNumericalTapTransfer> transfers() noexcept {
    return transfers_;
  }
  [[nodiscard]] std::span<const QwenNumericalTapTransfer> transfers()
      const noexcept {
    return transfers_;
  }
  [[nodiscard]] const QwenBf16TapTransferIdentity& identity() const noexcept {
    return identity_;
  }

 private:
  QwenBf16TapTransferPlan(
      QwenBf16TapTransferIdentity identity,
      std::vector<QwenNumericalTapTransfer> transfers)
      : identity_(identity), transfers_(std::move(transfers)) {}

  QwenBf16TapTransferIdentity identity_;
  std::vector<QwenNumericalTapTransfer> transfers_;
};

}  // namespace pih
