#include "sparse_attention.h"
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
Status ValidateSparseAttention(const SparseAttentionLaunch& x) {
  if (!x.stream || !x.tokens || x.tokens > 4096 || !x.heads || x.heads > 64 ||
      !x.kv_rows || x.kv_rows > 1048576 + 4096 || !x.picks || x.picks > 640)
    return Status::InvalidArgument("V4.1 sparse attention shape invalid");
  const auto query_bytes = static_cast<std::uint64_t>(x.tokens) * x.heads * 512 * 2;
  if (!Valid(x.query, query_bytes, 2) || !Valid(x.output, query_bytes, 2) ||
      !Valid(x.kv, x.kv_rows * 512ULL * 2, 2) || !Valid(x.sink, x.heads * 4ULL, 4) ||
      !Valid(x.indices, x.tokens * static_cast<std::uint64_t>(x.picks) * 4, 4) || !Valid(x.error_flag, 4, 4))
    return Status::InvalidArgument("V4.1 sparse attention buffer layout invalid");
  for (const auto input : {x.query, x.kv, x.sink, x.indices})
    if (Overlap(input, x.output) || Overlap(input, x.error_flag))
      return Status::InvalidArgument("V4.1 sparse attention writable/input alias");
  if (Overlap(x.output, x.error_flag)) return Status::InvalidArgument("V4.1 attention output/error alias");
  return Status::Ok();
}
}  // namespace pih::deepseek_v41
