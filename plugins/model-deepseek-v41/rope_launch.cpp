#include "rope_launch.h"
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
Status ValidateRopeTable(const RopeTableLaunch& x) {
  if (!x.stream || !x.tokens || x.tokens > 4096 || x.layer >= 43 ||
      !Valid(x.positions, x.tokens * 4ULL, 4) || !Valid(x.output, x.tokens * 64ULL * 4, 4) ||
      !Valid(x.error_flag, 4, 4) || Overlap(x.positions, x.output) ||
      Overlap(x.positions, x.error_flag) || Overlap(x.output, x.error_flag))
    return Status::InvalidArgument("V4.1 RoPE phase table contract invalid");
  return Status::Ok();
}
Status ValidateRopeApply(const RopeApplyLaunch& x) {
  if (!x.stream || !x.tokens || x.tokens > 4096 || !x.heads || x.heads > 64 ||
      (x.width != 128 && x.width != 512))
    return Status::InvalidArgument("V4.1 RoPE vector shape invalid");
  const auto bytes = static_cast<std::uint64_t>(x.tokens) * x.heads * x.width * 2;
  if (!Valid(x.input, bytes, 2) || !Valid(x.output, bytes, 2) ||
      !Valid(x.phases, x.tokens * 64ULL * 4, 4) || !Valid(x.error_flag, 4, 4) ||
      (Overlap(x.input, x.output) && x.input.address != x.output.address) ||
      Overlap(x.output, x.phases) || Overlap(x.error_flag, x.input) ||
      Overlap(x.error_flag, x.output) || Overlap(x.error_flag, x.phases))
    return Status::InvalidArgument("V4.1 RoPE buffer length, alignment or alias invalid");
  return Status::Ok();
}
Status ValidateRopeSequence(const RopeSequenceLaunch& x) {
  const auto table = ValidateRopeTable(x.table); if (!table.ok()) return table;
  if ((x.stride != 1 && x.stride != 2) || x.first_position >= 1048576 ||
      x.first_position + std::uint64_t(x.table.tokens - 1) * x.stride >= 1048576)
    return Status::InvalidArgument("RoPE position sequence exceeds supported original-position bounds");
  return Status::Ok();
}
}  // namespace pih::deepseek_v41
