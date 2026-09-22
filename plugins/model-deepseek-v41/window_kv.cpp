#include "window_kv.h"
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
Status ValidateWindowKv(const WindowKvLaunch& x) {
  if (!x.stream || !x.tokens || x.tokens > 4096 || (x.start && x.tokens != 1) ||
      x.start >= 1048576 || x.tokens > 1048576 - x.start ||
      !Valid(x.kv, x.tokens * 512ULL * 2, 2) || !Valid(x.ring, 128 * 512 * 2, 2) ||
      !Valid(x.error_flag, 4, 4) || Overlap(x.kv, x.ring) || Overlap(x.kv, x.error_flag) ||
      Overlap(x.ring, x.error_flag))
    return Status::InvalidArgument("V4.1 window KV sequence, buffer or alias invalid");
  return Status::Ok();
}
Status ValidateWindowKvPrepare(const WindowKvPrepareLaunch& x) {
  const auto norm = ValidateRmsNorm(x.norm); if (!norm.ok()) return norm;
  const auto rope = ValidateRopeApply(x.rope); if (!rope.ok()) return rope;
  const auto cache = ValidateWindowKv(x.cache); if (!cache.ok()) return cache;
  if (x.norm.width != 512 || x.rope.width != 512 || x.rope.heads != 1 || x.rope.inverse ||
      x.norm.rows != x.cache.tokens || x.rope.tokens != x.cache.tokens ||
      x.norm.stream != x.cache.stream || x.rope.stream != x.cache.stream ||
      !Same(x.norm.output, x.rope.input) || !Same(x.rope.output, x.cache.kv) ||
      !Same(x.norm.error_flag, x.cache.error_flag) || !Same(x.rope.error_flag, x.cache.error_flag))
    return Status::InvalidArgument("V4.1 window preparation stage connection invalid");
  const std::array reads{x.norm.input, x.norm.weight, x.rope.phases};
  const std::array writes{x.norm.output, x.rope.output, x.cache.ring, x.cache.error_flag};
  for (std::size_t i = 0; i < writes.size(); ++i) {
    for (const auto& read : reads)
      if (Overlap(writes[i], read)) return Status::InvalidArgument("Window preparation overwrites a live input");
    for (std::size_t j = 0; j < i; ++j)
      if (Overlap(writes[i], writes[j]) && !(i == 1 && j == 0 && Same(writes[i], writes[j])))
        return Status::InvalidArgument("Window preparation writable buffers overlap");
  }
  return Status::Ok();
}
Status LaunchWindowKvPrepare(const WindowKvPrepareLaunch& x) {
  const auto validation = ValidateWindowKvPrepare(x); if (!validation.ok()) return validation;
  const auto norm = LaunchRmsNorm(x.norm); if (!norm.ok()) return norm;
  const auto rope = LaunchRopeApply(x.rope); if (!rope.ok()) return rope;
  return LaunchWindowKv(x.cache);
}
}  // namespace pih::deepseek_v41
