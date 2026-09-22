#pragma once

#include <array>
#include <optional>
#include <utility>

#include "pih/backend/cuda/registered_pinned_allocator.h"
#include "pih/core/buffer.h"
#include "pih/model/deepseek_attention_output_projection_coordinator.h"
#include "pih/model/deepseek_attention_projection_coordinator.h"
#include "pih/model/deepseek_mhc_sequence_executor.h"
#include "pih/model/deepseek_v4_config.h"

namespace pih {

struct DeepSeekDenseMhcRuntimeOperations final {
  DeepSeekAttentionProjectionOperations* input = nullptr;
  DeepSeekAttentionOutputProjectionOperations* output = nullptr;
  DeepSeekMhcSequenceOperations* mhc = nullptr;
};

struct DeepSeekDenseMhcLayerRuntimeBorrow final {
  DeepSeekAttentionProjectionCoordinator* input = nullptr;
  DeepSeekAttentionOutputProjectionCoordinator* output = nullptr;
  DeepSeekMhcSequenceExecutor* mhc_attention = nullptr;
  DeepSeekMhcSequenceExecutor* mhc_feed_forward = nullptr;
};

class DeepSeekDenseMhcRuntimeResources final {
 public:
  static constexpr std::uint64_t kAlignment = 256;
  static Result<DeepSeekDenseMhcRuntimeResources> Allocate(
      DeepSeekStageRange owned_layers,
      DeepSeekDenseMhcRuntimeOperations operations,
      RegisteredPinnedAllocator& allocator);

  Result<DeepSeekDenseMhcLayerRuntimeBorrow> borrow(
      std::uint32_t layer) noexcept;
  [[nodiscard]] std::uint32_t layer_count() const noexcept {
    return layer_count_;
  }

 private:
  explicit DeepSeekDenseMhcRuntimeResources(Buffer host_errors) noexcept
      : host_errors_(std::move(host_errors)) {}
  Buffer host_errors_;
  DeepSeekStageRange owned_layers_;
  std::array<std::optional<DeepSeekAttentionProjectionCoordinator>, 43>
      input_;
  std::array<std::optional<DeepSeekAttentionOutputProjectionCoordinator>, 43>
      output_;
  std::array<std::optional<DeepSeekMhcSequenceExecutor>, 43> mhc_attention_;
  std::array<std::optional<DeepSeekMhcSequenceExecutor>, 43>
      mhc_feed_forward_;
  std::uint32_t layer_count_ = 0;
};

}  // namespace pih
