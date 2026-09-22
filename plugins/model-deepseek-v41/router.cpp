#include "router.h"
#include <limits>

namespace pih::deepseek_v41 {
namespace {
bool Valid(EngramDeviceRegion x, std::uint64_t bytes, unsigned alignment) {
  if (!bytes) return !x.address && !x.bytes;
  return x.address && x.address % alignment == 0 && x.bytes == bytes && bytes <= std::numeric_limits<std::uintptr_t>::max() - x.address;
}
bool Overlap(EngramDeviceRegion a, EngramDeviceRegion b) {
  return a.bytes && b.bytes && a.address < b.address + b.bytes && b.address < a.address + a.bytes;
}
}
Status ValidateRouter(const RouterLaunch& x) {
  if (!x.stream || !x.tokens || x.tokens > 4096 || x.layer >= 43)
    return Status::InvalidArgument("V4.1 router layer/token geometry invalid");
  const unsigned experts = x.layer < 40 ? 384 : 128, picks = x.layer < 40 ? 6 : 3;
  const bool image = x.image_mask.address || x.image_mask.bytes;
  if (!Valid(x.input, x.tokens * 5120ULL * 2, 2) || !Valid(x.weight, experts * 5120ULL * 4, 4) ||
      !Valid(x.bias, experts * 4, 4) || !Valid(x.image_mask, image ? x.tokens : 0, 1) ||
      !Valid(x.image_bias, image ? experts * 4 : 0, 4) || !Valid(x.logits, x.tokens * std::uint64_t(experts) * 4, 4) ||
      !Valid(x.indices, x.tokens * std::uint64_t(picks) * 4, 4) || !Valid(x.route_weights, x.tokens * std::uint64_t(picks) * 4, 4) ||
      !Valid(x.error_flag, 4, 4)) return Status::InvalidArgument("V4.1 router buffer layout invalid");
  const std::array reads{x.input, x.weight, x.bias, x.image_mask, x.image_bias};
  const std::array writes{x.logits, x.indices, x.route_weights, x.error_flag};
  for (std::size_t i = 0; i < writes.size(); ++i) {
    for (const auto read : reads) if (Overlap(read, writes[i])) return Status::InvalidArgument("Router overwrites live input");
    for (std::size_t j = 0; j < i; ++j) if (Overlap(writes[i], writes[j])) return Status::InvalidArgument("Router writable alias");
  }
  return Status::Ok();
}
}  // namespace pih::deepseek_v41
