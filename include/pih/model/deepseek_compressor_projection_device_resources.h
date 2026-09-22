#pragma once

#include "pih/core/buffer.h"

namespace pih {

struct DeepSeekCompressorProjectionSlice final {
  std::uintptr_t main_kv_f32 = 0;
  std::uintptr_t main_gate_f32 = 0;
  std::uintptr_t main_output_f32 = 0;
  std::uintptr_t index_kv_f32 = 0;
  std::uintptr_t index_gate_f32 = 0;
  std::uintptr_t index_output_f32 = 0;
  std::uintptr_t index_weights_f32 = 0;
};

class DeepSeekCompressorProjectionDeviceResources final {
 public:
  static Result<DeepSeekCompressorProjectionDeviceResources> Allocate(
      Allocator& allocator, std::uint32_t maximum_queries,
      std::uint64_t context_identity, std::int32_t device_ordinal);

  DeepSeekCompressorProjectionDeviceResources(
      const DeepSeekCompressorProjectionDeviceResources&) = delete;
  DeepSeekCompressorProjectionDeviceResources& operator=(
      const DeepSeekCompressorProjectionDeviceResources&) = delete;
  DeepSeekCompressorProjectionDeviceResources(
      DeepSeekCompressorProjectionDeviceResources&&) noexcept = default;
  DeepSeekCompressorProjectionDeviceResources& operator=(
      DeepSeekCompressorProjectionDeviceResources&&) noexcept = default;

  Result<DeepSeekCompressorProjectionSlice> slice(
      std::uint32_t query_ordinal) const;
  [[nodiscard]] std::uint32_t maximum_queries() const noexcept {
    return maximum_queries_;
  }

 private:
  DeepSeekCompressorProjectionDeviceResources(
      Buffer storage, std::uint32_t maximum_queries,
      std::uint64_t context_identity, std::int32_t device_ordinal) noexcept
      : storage_(std::move(storage)), maximum_queries_(maximum_queries),
        context_identity_(context_identity), device_ordinal_(device_ordinal) {}

  Buffer storage_;
  std::uint32_t maximum_queries_ = 0;
  std::uint64_t context_identity_ = 0;
  std::int32_t device_ordinal_ = -1;
};

}  // namespace pih
