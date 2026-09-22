#pragma once

#include <cstdint>
#include <vector>

#include "pih/core/buffer.h"
#include "pih/model/deepseek_expert_compute_arena.h"
#include "pih/model/deepseek_expert_slot_table.h"
#include "pih/model/deepseek_shared_expert_driver.h"

namespace pih {

class DeepSeekRankExpertDeviceResources final {
 public:
  static Result<DeepSeekRankExpertDeviceResources> Allocate(
      Allocator& allocator, std::uint32_t slot_count,
      std::uint32_t maximum_tokens, std::uint64_t context_identity,
      std::int32_t device_ordinal,
      Allocator* host_spill_slot_allocator = nullptr);

  DeepSeekRankExpertDeviceResources(
      const DeepSeekRankExpertDeviceResources&) = delete;
  DeepSeekRankExpertDeviceResources& operator=(
      const DeepSeekRankExpertDeviceResources&) = delete;
  DeepSeekRankExpertDeviceResources(
      DeepSeekRankExpertDeviceResources&&) noexcept = default;
  DeepSeekRankExpertDeviceResources& operator=(
      DeepSeekRankExpertDeviceResources&&) noexcept = default;

  [[nodiscard]] const std::vector<std::uintptr_t>& slot_bases() const noexcept {
    return slot_bases_;
  }
  [[nodiscard]] const DeepSeekExpertSlotTable& slots() const noexcept {
    return slots_;
  }
  [[nodiscard]] DeepSeekExpertSlotTable slot_table() const { return slots_; }
  [[nodiscard]] DeepSeekExpertComputeArena arena() const noexcept {
    return arena_;
  }
  // Borrowed views: use only after ALL routed experts have completed and before
  // launching the next layer. Shared output reuses routed input/output storage;
  // source_hidden_ and accumulator_ remain separate allocations.
  [[nodiscard]] DeepSeekSharedExpertArena shared_arena() const noexcept {
    return {arena_.activation_e4m3, arena_.activation_scale_bits,
            arena_.gate_or_middle_bf16, arena_.up_bf16,
            arena_.expert_output_bf16, arena_.error_flag_u32};
  }
  [[nodiscard]] std::uint32_t maximum_tokens() const noexcept {
    return maximum_tokens_;
  }
  [[nodiscard]] std::uintptr_t source_hidden_bf16() const noexcept {
    return reinterpret_cast<std::uintptr_t>(source_hidden_.data());
  }
  [[nodiscard]] std::uintptr_t accumulator_f32() const noexcept {
    return reinterpret_cast<std::uintptr_t>(accumulator_.data());
  }
  [[nodiscard]] std::uintptr_t compute_backing_address() const noexcept {
    return reinterpret_cast<std::uintptr_t>(compute_backing_.data());
  }
  [[nodiscard]] std::uint64_t compute_backing_bytes() const noexcept {
    return compute_backing_.size_bytes();
  }
  [[nodiscard]] std::uint64_t source_hidden_bytes() const noexcept {
    return source_hidden_.size_bytes();
  }
  [[nodiscard]] std::uint64_t accumulator_bytes() const noexcept {
    return accumulator_.size_bytes();
  }

 private:
  DeepSeekRankExpertDeviceResources(
      std::vector<Buffer> slot_backings,
      std::vector<std::uintptr_t> slot_bases, DeepSeekExpertSlotTable slots,
      Buffer compute_backing, DeepSeekExpertComputeArena arena,
      Buffer source_hidden, Buffer accumulator, std::uint32_t maximum_tokens)
      : slot_backings_(std::move(slot_backings)),
        slot_bases_(std::move(slot_bases)), slots_(std::move(slots)),
        compute_backing_(std::move(compute_backing)), arena_(arena),
        source_hidden_(std::move(source_hidden)),
        accumulator_(std::move(accumulator)), maximum_tokens_(maximum_tokens) {}

  std::vector<Buffer> slot_backings_;
  std::vector<std::uintptr_t> slot_bases_;
  DeepSeekExpertSlotTable slots_;
  Buffer compute_backing_;
  DeepSeekExpertComputeArena arena_;
  Buffer source_hidden_;
  Buffer accumulator_;
  std::uint32_t maximum_tokens_ = 0;
};

}  // namespace pih
