#include "linear_fp8.h"
#include <limits>

namespace pih::deepseek_v41 {
namespace {
bool Valid(EngramDeviceRegion x, std::uint64_t bytes, unsigned alignment) {
  return x.address && x.address % alignment == 0 && x.bytes == bytes &&
      bytes <= std::numeric_limits<std::uintptr_t>::max() - x.address;
}
bool Overlap(EngramDeviceRegion a, EngramDeviceRegion b) {
  return a.address < b.address + b.bytes && b.address < a.address + a.bytes;
}
}
Status ValidateFp8Linear(const Fp8LinearLaunch& x) {
  if (!x.stream || !x.rows || x.rows > 4096 || !x.in_features || x.in_features > 32768 ||
      !x.out_features || x.out_features > 32768 || x.in_features % 32 || x.out_features % 32)
    return Status::InvalidArgument("V4.1 FP8 linear requires bounded block-32 dimensions");
  const std::uint64_t m = x.rows, k = x.in_features, n = x.out_features;
  if (!Valid(x.input, m * k * 2, 2) || !Valid(x.weight, n * k, 1) ||
      !Valid(x.weight_scales, n / 32 * (k / 32), 1) || !Valid(x.quantized, m * k, 1) ||
      !Valid(x.activation_scales, m * (k / 32), 1) || !Valid(x.output, m * n * 2, 2) ||
      !Valid(x.error_flag, 4, 4)) return Status::InvalidArgument("V4.1 FP8 linear buffer layout invalid");
  const std::array inputs{x.input, x.weight, x.weight_scales};
  const std::array outputs{x.quantized, x.activation_scales, x.output, x.error_flag};
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    for (const auto input : inputs)
      if (Overlap(input, outputs[i])) return Status::InvalidArgument("FP8 linear overwrites an input");
    for (std::size_t j = 0; j < i; ++j)
      if (Overlap(outputs[i], outputs[j])) return Status::InvalidArgument("FP8 linear writable alias");
  }
  return Status::Ok();
}
}  // namespace pih::deepseek_v41
