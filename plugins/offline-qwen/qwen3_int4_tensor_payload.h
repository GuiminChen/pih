#pragma once
// Private to native offline Qwen conversion; not a model/runtime SDK interface.

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "pih/core/result.h"
#include "pih/core/sha256.h"
#include "pih/model/qwen3_int4_linear_shape_ledger.h"

namespace pih {

class QwenInt4TensorPayload final {
 public:
  static Result<QwenInt4TensorPayload> Create(
      std::span<const std::byte> source_bf16_le,
      const QwenInt4LinearRecordPlan& plan);

  [[nodiscard]] std::span<const std::byte> packed_values() const noexcept {
    return packed_values_;
  }
  [[nodiscard]] std::span<const std::byte> scale_bytes_le() const noexcept {
    return scale_bytes_le_;
  }
  [[nodiscard]] const Sha256Digest& packed_sha256() const noexcept {
    return packed_sha256_;
  }
  [[nodiscard]] const Sha256Digest& scales_sha256() const noexcept {
    return scales_sha256_;
  }
  [[nodiscard]] std::uint64_t peak_anonymous_workspace_bytes() const noexcept {
    return peak_anonymous_workspace_bytes_;
  }
  [[nodiscard]] std::uint64_t retained_payload_bytes() const noexcept {
    return packed_values_.size() + scale_bytes_le_.size();
  }

 private:
  QwenInt4TensorPayload(std::vector<std::byte> packed,
                        std::vector<std::byte> scales,
                        Sha256Digest packed_sha256,
                        Sha256Digest scales_sha256,
                        std::uint64_t peak)
      : packed_values_(std::move(packed)), scale_bytes_le_(std::move(scales)),
        packed_sha256_(packed_sha256), scales_sha256_(scales_sha256),
        peak_anonymous_workspace_bytes_(peak) {}

  std::vector<std::byte> packed_values_;
  std::vector<std::byte> scale_bytes_le_;
  Sha256Digest packed_sha256_;
  Sha256Digest scales_sha256_;
  std::uint64_t peak_anonymous_workspace_bytes_;
};

}  // namespace pih
