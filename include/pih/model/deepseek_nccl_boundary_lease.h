#pragma once

#include "pih/core/buffer.h"
#include "pih/model/deepseek_nccl_p2p_plan.h"

namespace pih {

class DeepSeekNcclBoundaryLease final {
 public:
  static Result<DeepSeekNcclBoundaryLease> Create(
      const DeepSeekNcclP2pManifest& manifest, Buffer& buffer,
      std::uint64_t owner_id, std::uintptr_t context_identity,
      std::uintptr_t boundary_stream);
  Status bind(DeepSeekNcclP2pBindingTarget& target) const;
  [[nodiscard]] void* data() const noexcept { return data_; }
  [[nodiscard]] std::uint64_t bytes() const noexcept { return bytes_; }
  [[nodiscard]] std::uintptr_t stream() const noexcept { return stream_; }

 private:
  DeepSeekNcclBoundaryLease(DeepSeekNcclRole role, void* data,
                            std::uint64_t bytes, std::uintptr_t stream) noexcept
      : role_(role), data_(data), bytes_(bytes), stream_(stream) {}
  DeepSeekNcclRole role_;
  void* data_ = nullptr;
  std::uint64_t bytes_ = 0;
  std::uintptr_t stream_ = 0;
};

}  // namespace pih
