#include "norm_launch.h"
#include <limits>

namespace pih::deepseek_v41 {
namespace {
bool Valid(EngramDeviceRegion region, std::uint64_t bytes, unsigned alignment) {
  return region.address && region.address % alignment == 0 && region.bytes == bytes &&
      bytes <= std::numeric_limits<std::uintptr_t>::max() - region.address;
}
bool Overlap(EngramDeviceRegion a, EngramDeviceRegion b) {
  return a.address < b.address + b.bytes && b.address < a.address + a.bytes;
}
}
Status ValidateRmsNorm(const RmsNormLaunch& x) {
  if (!x.stream || !x.rows || x.rows > 4096 ||
      (x.width != 128 && x.width != 512 && x.width != 1280 && x.width != 5120) ||
      (x.weight_storage != EngramStorage::kBF16 && x.weight_storage != EngramStorage::kF32))
    return Status::InvalidArgument("V4.1 RMSNorm shape or storage invalid");
  const unsigned weight_bytes = x.weight_storage == EngramStorage::kBF16 ? 2 : 4;
  const auto bytes = static_cast<std::uint64_t>(x.rows) * x.width * 2;
  if (!Valid(x.input, bytes, 2) || !Valid(x.output, bytes, 2) ||
      !Valid(x.weight, static_cast<std::uint64_t>(x.width) * weight_bytes, weight_bytes) ||
      !Valid(x.error_flag, 4, 4) || Overlap(x.output, x.input) || Overlap(x.output, x.weight) ||
      Overlap(x.output, x.error_flag) || Overlap(x.error_flag, x.input) || Overlap(x.error_flag, x.weight))
    return Status::InvalidArgument("V4.1 RMSNorm buffer size, alignment or alias invalid");
  return Status::Ok();
}
}  // namespace pih::deepseek_v41
