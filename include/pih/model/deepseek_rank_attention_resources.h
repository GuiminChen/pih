#pragma once

#include <memory>

#include "pih/model/deepseek_attention_device_scratch_resources.h"
#include "pih/model/deepseek_attention_host_staging_resources.h"
#include "pih/model/deepseek_compressor_projection_device_resources.h"

namespace pih {

class DeepSeekRankAttentionResources final {
 public:
  static Result<DeepSeekRankAttentionResources> Allocate(
      DeepSeekStageRange owned_layers, std::uint32_t maximum_queries,
      std::uint32_t maximum_ratio4_logical_pages,
      DeepSeekFixedStateLayout fixed_layout,
      DeepSeekAttentionPageArena page_arena,
      DeepSeekAttentionRuntimeOperations operations,
      RegisteredPinnedAllocator& pinned_allocator,
      Allocator& device_allocator, std::uintptr_t completion_event,
      std::uint64_t context_identity, std::int32_t device_ordinal);

  DeepSeekRankAttentionResources(
      const DeepSeekRankAttentionResources&) = delete;
  DeepSeekRankAttentionResources& operator=(
      const DeepSeekRankAttentionResources&) = delete;
  DeepSeekRankAttentionResources(
      DeepSeekRankAttentionResources&&) noexcept = default;
  DeepSeekRankAttentionResources& operator=(
      DeepSeekRankAttentionResources&& other) noexcept;

  [[nodiscard]] DeepSeekAttentionWorkFactory& work_factory() noexcept {
    return *work_factory_;
  }
  [[nodiscard]] DeepSeekAttentionDeviceScratchResources& device_scratch()
      noexcept { return *device_scratch_; }
  [[nodiscard]] DeepSeekCompressorProjectionDeviceResources&
  compressor_projection() noexcept { return *compressor_projection_; }
  [[nodiscard]] std::uint32_t maximum_queries() const noexcept {
    return host_staging_->maximum_queries();
  }

 private:
  DeepSeekRankAttentionResources(
      std::unique_ptr<DeepSeekAttentionHostStagingResources> host_staging,
      std::unique_ptr<DeepSeekAttentionDeviceScratchResources> device_scratch,
      std::unique_ptr<DeepSeekCompressorProjectionDeviceResources>
          compressor_projection,
      std::unique_ptr<DeepSeekAttentionRuntimeResources> runtime,
      std::unique_ptr<DeepSeekAttentionWorkFactory> work_factory) noexcept
      : host_staging_(std::move(host_staging)),
        device_scratch_(std::move(device_scratch)),
        compressor_projection_(std::move(compressor_projection)),
        runtime_(std::move(runtime)), work_factory_(std::move(work_factory)) {}

  // Factory borrows runtime; runtime borrows staging and external operations.
  std::unique_ptr<DeepSeekAttentionHostStagingResources> host_staging_;
  std::unique_ptr<DeepSeekAttentionDeviceScratchResources> device_scratch_;
  std::unique_ptr<DeepSeekCompressorProjectionDeviceResources>
      compressor_projection_;
  std::unique_ptr<DeepSeekAttentionRuntimeResources> runtime_;
  std::unique_ptr<DeepSeekAttentionWorkFactory> work_factory_;
};

}  // namespace pih
