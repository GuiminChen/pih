#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "pih/backend/cuda/typed_copy_plan.h"
#include "pih/model/qwen3_bf16_resource_set.h"
#include "pih/model/qwen3_bf16_tap_binding.h"
#include "pih/model/qwen3_kv_address_mapper.h"
#include "pih/model/qwen3_kv_block_table.h"

namespace pih {

class QwenBf16TapSourcePlan final {
 public:
  static Result<QwenBf16TapSourcePlan> Create(
      const QwenNumericalTapPlan& taps,
      const QwenBf16TapBindingPlan& bindings,
      const QwenBf16ResourceSet& resources,
      const QwenKvBlockTable& block_table,
      std::span<const QwenKvSlotState> slot_states,
      std::uint64_t activation_first_position,
      std::uint64_t source_owner_id_base);

  [[nodiscard]] std::size_t size() const noexcept { return sources_.size(); }
  [[nodiscard]] const CudaCopyEndpoint& operator[](
      std::size_t index) const noexcept {
    return sources_[index];
  }
  [[nodiscard]] std::span<const CudaCopyEndpoint> sources() const noexcept {
    return sources_;
  }

 private:
  explicit QwenBf16TapSourcePlan(std::vector<CudaCopyEndpoint> sources)
      : sources_(std::move(sources)) {}

  std::vector<CudaCopyEndpoint> sources_;
};

}  // namespace pih
