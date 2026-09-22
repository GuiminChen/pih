#pragma once
#include "pih/core/result.h"
#include <cstddef>
#include <cstdint>
#include <span>
#include <functional>

namespace pih::deepseek_v41 {
// Canonical positive E8M0 scale; 255 is NaN, zero encodes 2^-127, not zero.
Result<float> DecodeWeightE8M0(std::uint8_t bits);
// One row-major 32x32 or 128x128 E4M3FN source block and its numeric scale.
// Implements the reference wo_a FP32 multiply then BF16 round-to-nearest-even.
// Output is explicitly little-endian BF16. Both spans must have exact lengths.
// Entire block is validated before output changes; input/output may overlap.
// Reject nonfinite inputs/results and unsupported floating-point environment.
// No source naming, shape catalog, file admission or publication is implied.
Status DequantizeWoABlock(std::span<const std::byte> source, float scale,
    std::uint32_t block_size, std::span<std::byte> destination);

enum class WoAScaleStorage { kE8M0, kF32LittleEndian };
struct WoASourceGeometry final {
  std::uint64_t rows = 0, columns = 0;
  std::uint64_t scale_rows = 0, scale_columns = 0;
  std::uint64_t weight_bytes = 0, scale_bytes = 0;
  WoAScaleStorage scale_storage{};
};
using WoATensorReader = std::function<Status(std::uint64_t, std::span<std::byte>)>;
using WoATensorWriter = std::function<Status(std::uint64_t, std::span<const std::byte>)>;
// Whole unpartitioned backbone wo_a: E4M3FN [8192,4096], scale blocks 32/128.
// Exact-span callbacks use tensor-relative byte offsets. Emits consecutive BF16
// row bands of at most 1 MiB, after validating all values in each band. Failure
// invalidates the entire unpublished output, even if earlier bands were written.
// Scratch is bounded to <2 MiB; no full source/output tensor is allocated.
Status DequantizeWoATensor(const WoASourceGeometry& geometry,
    const WoATensorReader& read_weight, const WoATensorReader& read_scale,
    const WoATensorWriter& write);
}  // namespace pih::deepseek_v41
