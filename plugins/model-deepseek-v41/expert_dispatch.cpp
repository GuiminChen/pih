#include "expert_dispatch.h"
#include <limits>

namespace pih::deepseek_v41 {
namespace {
bool Valid(EngramDeviceRegion x, std::uint64_t bytes) {
  return x.address && x.address % 4 == 0 && x.bytes == bytes && bytes <= std::numeric_limits<std::uintptr_t>::max() - x.address;
}
bool Overlap(EngramDeviceRegion a, EngramDeviceRegion b) {
  return a.address < b.address + b.bytes && b.address < a.address + a.bytes;
}
}
Status ValidateExpertDispatch(const ExpertDispatchLaunch& x) {
  if (!x.stream || !x.tokens || x.tokens > 4096 || x.layer >= 43 ||
      (x.world_size != 1 && x.world_size != 2 && x.world_size != 4 && x.world_size != 8) || x.rank >= x.world_size)
    return Status::InvalidArgument("V4.1 expert dispatch geometry invalid");
  const unsigned experts = x.layer < 40 ? 384 : 128, picks = x.layer < 40 ? 6 : 3;
  const unsigned local = experts / x.world_size;
  if (!Valid(x.indices, x.tokens * std::uint64_t(picks) * 4) || !Valid(x.counts, local * 4) ||
      !Valid(x.slots, local * std::uint64_t(x.tokens) * 4) || !Valid(x.error_flag, 4))
    return Status::InvalidArgument("V4.1 expert dispatch buffer layout invalid");
  const std::array writes{x.counts, x.slots, x.error_flag};
  for (std::size_t i = 0; i < writes.size(); ++i) {
    if (Overlap(x.indices, writes[i])) return Status::InvalidArgument("Expert dispatch overwrites router indices");
    for (std::size_t j = 0; j < i; ++j) if (Overlap(writes[i], writes[j])) return Status::InvalidArgument("Expert dispatch writable alias");
  }
  return Status::Ok();
}
Status ValidateExpertGather(const ExpertGatherLaunch& x) {
  const auto dispatch = ValidateExpertDispatch(x.dispatch); if (!dispatch.ok()) return dispatch;
  const auto& d = x.dispatch;
  const unsigned experts = d.layer < 40 ? 384 : 128, picks = d.layer < 40 ? 6 : 3;
  const unsigned local = experts / d.world_size, first = d.rank * local;
  const auto region = [](EngramDeviceRegion r, std::uint64_t bytes, unsigned alignment) {
    if (!bytes) return !r.address && !r.bytes;
    return r.address && r.address % alignment == 0 && r.bytes == bytes && bytes <= std::numeric_limits<std::uintptr_t>::max() - r.address;
  };
  if (x.expert < first || x.expert >= first + local || x.rows > d.tokens ||
      !region(x.input, d.tokens * 5120ULL * 2, 2) || !region(x.route_weights, d.tokens * std::uint64_t(picks) * 4, 4) ||
      !region(x.output, x.rows * 5120ULL * 2, 2) || !region(x.gathered_weights, x.rows * 4ULL, 4))
    return Status::InvalidArgument("V4.1 expert gather ownership, rows or layout invalid");
  const std::array reads{d.indices, d.counts, d.slots, x.input, x.route_weights};
  const std::array writes{x.output, x.gathered_weights, d.error_flag};
  for (std::size_t i = 0; i < writes.size(); ++i) {
    for (const auto read : reads) if (writes[i].bytes && Overlap(read, writes[i])) return Status::InvalidArgument("Expert gather overwrites live input");
    for (std::size_t j = 0; j < i; ++j)
      if (writes[i].bytes && writes[j].bytes && Overlap(writes[i], writes[j])) return Status::InvalidArgument("Expert gather writable alias");
  }
  return Status::Ok();
}
Status ValidateExpertScatter(const ExpertScatterLaunch& x) {
  const auto dispatch = ValidateExpertDispatch(x.dispatch); if (!dispatch.ok()) return dispatch;
  const auto& d = x.dispatch;
  const unsigned experts = d.layer < 40 ? 384 : 128;
  const unsigned local = experts / d.world_size, first = d.rank * local;
  const std::uint64_t input_bytes = x.rows * 5120ULL * 2;
  const bool input_valid = input_bytes
      ? x.input.address && x.input.address % 2 == 0 && x.input.bytes == input_bytes &&
          input_bytes <= std::numeric_limits<std::uintptr_t>::max() - x.input.address
      : !x.input.address && !x.input.bytes;
  if (x.expert < first || x.expert >= first + local || x.rows > d.tokens || !input_valid ||
      !Valid(x.accumulator, d.tokens * 5120ULL * 4))
    return Status::InvalidArgument("V4.1 expert scatter ownership, rows or layout invalid");
  const std::array reads{d.indices, d.counts, d.slots, x.input};
  const std::array writes{x.accumulator, d.error_flag};
  for (const auto write : writes) for (const auto read : reads)
    if (read.bytes && Overlap(write, read)) return Status::InvalidArgument("Expert scatter overwrites live input");
  if (Overlap(x.accumulator, d.error_flag)) return Status::InvalidArgument("Expert scatter writable alias");
  return Status::Ok();
}
}  // namespace pih::deepseek_v41
