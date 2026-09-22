#pragma once

#include <memory>
#include <span>

#include "pih/backend/cuda/registered_pinned_allocator.h"
#include "pih/model/deepseek_attention_projection_device_resources.h"

namespace pih {

class DeepSeekRequestInputCopyOperations {
 public:
  virtual ~DeepSeekRequestInputCopyOperations() = default;
  virtual Status copy_h2d_async(std::uintptr_t destination,
                                const void* source, std::uint64_t bytes,
                                std::uintptr_t stream) = 0;
  virtual Status synchronize(std::uintptr_t stream) = 0;
};

struct DeepSeekRequestInputStagingLease final {
  std::shared_ptr<Buffer> owner;
  std::uint32_t token_count = 0;
};

class DeepSeekRequestInputStagingResources final {
 public:
  static Result<DeepSeekRequestInputStagingResources> Allocate(
      std::uint32_t maximum_tokens, RegisteredPinnedAllocator& allocator);

  Result<DeepSeekRequestInputStagingLease> stage(
      std::span<const std::uint32_t> token_ids,
      std::span<const std::uint32_t> positions,
      DeepSeekAttentionProjectionDeviceView destination,
      std::uintptr_t stream,
      DeepSeekRequestInputCopyOperations& operations);

  [[nodiscard]] std::uint32_t maximum_tokens() const noexcept {
    return maximum_tokens_;
  }

  // A failed synchronization keeps the engine graph quarantined. Retrying
  // close may retire it only after the native provider proves completion.
  Status retire_pending() noexcept;

 private:
  DeepSeekRequestInputStagingResources(
      std::shared_ptr<Buffer> backing, std::uint64_t positions_offset,
      std::uint32_t maximum_tokens) noexcept
      : backing_(std::move(backing)), positions_offset_(positions_offset),
        maximum_tokens_(maximum_tokens) {}

  std::shared_ptr<Buffer> backing_;
  std::uint64_t positions_offset_ = 0;
  std::uint32_t maximum_tokens_ = 0;
  DeepSeekRequestInputCopyOperations* pending_operations_ = nullptr;
  std::uintptr_t pending_stream_ = 0;
};

}  // namespace pih
