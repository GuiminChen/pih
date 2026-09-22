#pragma once
#include "pih/core/result.h"
#include "weight_dequantize.h"
#include <array>
#include <cstddef>
#include <span>

namespace pih::deepseek_v41 {
struct ExpertFp8Block final {
  std::array<std::byte, 32 * 32> weight;
  std::byte scale;
};
// One logical 32x32 FP4 block: 32 rows of 16 packed bytes (low nibble first),
// plus one E8M0 scale per row. Output is E4M3FN with one shared E8M0 scale.
// Scale follows reference max(row_scale)/64; values round to nearest, ties even.
// Reject NaN scales or a shared scale below the E8M0 representable range.
// No file admission, source layout inference or losslessness claim is implied.
Result<ExpertFp8Block> ConvertExpertFp4Block(
    std::span<const std::byte> packed, std::span<const std::byte> row_scales);
// Backbone expert matrices only: logical [2304,5120] or [5120,2304].
// Input weight is row-major packed [rows,columns/2]; E8M0 scales are
// [rows,columns/32]. Produces FP8 [rows,columns] and E8M0 [rows/32,columns/32].
// Offsets are tensor-relative. Either writer failing invalidates BOTH outputs.
Status ConvertExpertFp4Tensor(std::uint64_t rows, std::uint64_t columns,
    const WoATensorReader& read_weight, const WoATensorReader& read_scales,
    const WoATensorWriter& write_weight, const WoATensorWriter& write_scales);
}  // namespace pih::deepseek_v41
