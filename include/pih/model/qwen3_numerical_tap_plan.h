#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "pih/core/dtype.h"
#include "pih/core/sha256.h"

namespace pih {

enum class QwenNumericalTapPoint : std::uint8_t {
  kLayerHidden = 0,
  kQueryAfterNorm,
  kKeyAfterNorm,
  kQueryAfterRope,
  kKeyAfterRope,
  kPrefillAttention,
  kKvKey,
  kKvValue,
  kFinalNorm,
  kLogits,
};

struct QwenNumericalTapRequest final {
  QwenNumericalTapPoint point;
  std::uint32_t layer;
  std::uint32_t position;
  std::uint32_t rows;
  friend bool operator==(const QwenNumericalTapRequest&,
                         const QwenNumericalTapRequest&) = default;
};

struct QwenNumericalTapCapture final {
  QwenNumericalTapRequest request;
  DType dtype;
  std::uint64_t element_count;
  std::uint64_t offset_bytes;
  std::uint64_t size_bytes;
};

class QwenNumericalTapPlan final {
 public:
  static constexpr std::size_t kMaximumCaptureCount = 512;
  static constexpr std::uint64_t kMaximumArenaBytes = 1ULL << 30;
  static constexpr std::uint64_t kAlignment = 256;

  static Result<QwenNumericalTapPlan> Create(
      std::span<const QwenNumericalTapRequest> requests,
      std::uint64_t arena_capacity_bytes);

  [[nodiscard]] std::span<const QwenNumericalTapCapture> captures() const noexcept {
    return captures_;
  }
  [[nodiscard]] std::uint64_t arena_bytes() const noexcept { return arena_bytes_; }
  Result<Sha256Digest> semantic_digest() const;

 private:
  QwenNumericalTapPlan(std::vector<QwenNumericalTapCapture> captures,
                       std::uint64_t arena_bytes)
      : captures_(std::move(captures)), arena_bytes_(arena_bytes) {}

  std::vector<QwenNumericalTapCapture> captures_;
  std::uint64_t arena_bytes_;
};

Status validate_qwen_bf16_tap_coverage(const QwenNumericalTapPlan& plan);

}  // namespace pih
