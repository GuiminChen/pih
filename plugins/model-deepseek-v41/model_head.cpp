#include "model_head.h"
#include <array>
#include <limits>

namespace pih::deepseek_v41 {
namespace {
bool Same(EngramDeviceRegion a, EngramDeviceRegion b) { return a.address == b.address && a.bytes == b.bytes; }
bool Overlap(EngramDeviceRegion a, EngramDeviceRegion b) {
  return a.bytes && b.bytes && a.address < b.address + b.bytes && b.address < a.address + a.bytes;
}
bool Valid(EngramDeviceRegion r, std::uint64_t bytes, unsigned alignment) {
  return r.address && r.address % alignment == 0 && r.bytes == bytes && bytes <= std::numeric_limits<std::uintptr_t>::max() - r.address;
}
}
Status ValidateHeadProjection(const HeadProjectionLaunch& x) {
  if (!x.stream || (x.world_size != 1 && x.world_size != 2 && x.world_size != 4 && x.world_size != 8) || x.rank >= x.world_size)
    return Status::InvalidArgument("Vocabulary head rank or stream invalid");
  const auto rows = 129280U / x.world_size;
  if (!Valid(x.input, 5120ULL * 2, 2) || !Valid(x.weight, rows * 5120ULL * 4, 4) ||
      !Valid(x.output, rows * 4ULL, 4) || !Valid(x.error_flag, 4, 4))
    return Status::InvalidArgument("Vocabulary head storage invalid");
  for (const auto write : {x.output, x.error_flag}) for (const auto read : {x.input, x.weight})
    if (Overlap(write, read)) return Status::InvalidArgument("Vocabulary head overwrites an input or weight");
  if (Overlap(x.output, x.error_flag)) return Status::InvalidArgument("Vocabulary head output/error alias");
  return Status::Ok();
}
Status ValidateModelHead(const ModelHeadLaunch& x) {
  const auto input = ValidateMhcInput(x.input); if (!input.ok()) return input;
  const auto projection = ValidateHeadProjection(x.projection); if (!projection.ok()) return projection;
  const auto& c = x.input.collapse; const auto& n = x.input.norm; const auto& p = x.projection;
  const EngramDeviceRegion last{n.output.address + (n.rows - 1ULL) * 5120 * 2, 5120ULL * 2};
  if (n.width != 5120 || !Same(last, p.input) || p.stream != c.stream || !Same(p.error_flag, c.error_flag))
    return Status::InvalidArgument("Model head must project the last normalized position on the same stream");
  const std::array reads{c.residual, c.pre, n.weight, p.weight};
  const std::array writes{c.output, n.output, p.output, c.error_flag};
  for (std::size_t i = 0; i < writes.size(); ++i) {
    for (const auto read : reads) if (Overlap(writes[i], read))
      return Status::InvalidArgument("Model head overwrites a live input or weight");
    for (std::size_t j = 0; j < i; ++j) if (Overlap(writes[i], writes[j]))
      return Status::InvalidArgument("Model head writable storage alias");
  }
  return Status::Ok();
}
Status LaunchModelHead(const ModelHeadLaunch& x) {
  const auto validation = ValidateModelHead(x); if (!validation.ok()) return validation;
  const auto input = LaunchMhcInput(x.input); if (!input.ok()) return input;
  return LaunchHeadProjection(x.projection);
}
}  // namespace pih::deepseek_v41
