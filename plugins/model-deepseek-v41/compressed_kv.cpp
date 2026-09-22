#include "compressed_kv.h"
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
bool Same(EngramDeviceRegion a, EngramDeviceRegion b) { return a.address == b.address && a.bytes == b.bytes; }
}
Status ValidateCompressedKv(const CompressedKvLaunch& x) {
  if (!x.stream || !x.rows || x.rows > 4096 || !x.capacity || x.capacity > 1048576 ||
      x.first_slot >= x.capacity || x.rows > x.capacity - x.first_slot ||
      !Valid(x.kv, x.rows * 512ULL * 2, 2) || !Valid(x.cache, x.capacity * 512ULL * 2, 2) ||
      !Valid(x.error_flag, 4, 4) || Overlap(x.kv, x.cache) || Overlap(x.kv, x.error_flag) ||
      Overlap(x.cache, x.error_flag))
    return Status::InvalidArgument("V4.1 compressed KV extent, buffer or alias invalid");
  return Status::Ok();
}
Status ValidateCompressedKvPrepare(const CompressedKvPrepareLaunch& x) {
  const auto rope = ValidateRopeApply(x.rope); if (!rope.ok()) return rope;
  const auto cache = ValidateCompressedKv(x.cache); if (!cache.ok()) return cache;
  if (x.rope.width != 512 || x.rope.heads != 1 || x.rope.inverse ||
      x.rope.tokens != x.cache.rows || x.rope.stream != x.cache.stream ||
      !Same(x.rope.output, x.cache.kv) || !Same(x.rope.error_flag, x.cache.error_flag))
    return Status::InvalidArgument("V4.1 compressed KV preparation connection invalid");
  for (const auto read : std::array{x.rope.input, x.rope.phases})
    for (const auto write : std::array{x.cache.kv, x.cache.cache, x.cache.error_flag})
      if (Overlap(read, write)) return Status::InvalidArgument("Compressed KV preparation overwrites a live input");
  return Status::Ok();
}
Status LaunchCompressedKvPrepare(const CompressedKvPrepareLaunch& x) {
  const auto validation = ValidateCompressedKvPrepare(x); if (!validation.ok()) return validation;
  const auto rope = LaunchRopeApply(x.rope); if (!rope.ok()) return rope;
  return LaunchCompressedKv(x.cache);
}
}  // namespace pih::deepseek_v41
