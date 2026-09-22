#include "indexer_select.h"
#include <algorithm>
#include <limits>

namespace pih::deepseek_v41 {
namespace {
bool Valid(EngramDeviceRegion x, std::uint64_t bytes, unsigned alignment) {
  if (!bytes) return !x.address && !x.bytes;
  return x.address && x.address % alignment == 0 && x.bytes == bytes &&
      bytes <= std::numeric_limits<std::uintptr_t>::max() - x.address;
}
bool Overlap(EngramDeviceRegion a, EngramDeviceRegion b) {
  return a.bytes && b.bytes && a.address < b.address + b.bytes && b.address < a.address + a.bytes;
}
}
Status ValidateIndexerSelect(const IndexerSelectLaunch& x) {
  if (!x.stream || !x.tokens || x.tokens > 4096 || (x.start && x.tokens != 1) ||
      x.start >= 1048576 || x.tokens > 1048576 - x.start || (x.ratio != 1 && x.ratio != 2) ||
      x.positions != (x.start + x.tokens) / x.ratio || x.offset > 4096)
    return Status::InvalidArgument("V4.1 index selection step geometry invalid");
  const auto picks = std::min(512U, x.positions);
  const bool candidates = x.candidates.address || x.candidates.bytes;
  if (candidates && (x.ratio != 1 || !Valid(x.candidates, x.tokens * std::uint64_t(x.positions), 1) ||
      Overlap(x.candidates, x.output) || Overlap(x.candidates, x.error_flag)))
    return Status::InvalidArgument("V4.1 index candidate mask layout or alias invalid");
  if (!Valid(x.scores, x.tokens * std::uint64_t(x.positions) * 2, 2) ||
      !Valid(x.output, x.tokens * std::uint64_t(picks) * 4, 4) || !Valid(x.error_flag, 4, 4) ||
      Overlap(x.scores, x.output) || Overlap(x.scores, x.error_flag) || Overlap(x.output, x.error_flag))
    return Status::InvalidArgument("V4.1 index selection buffer or alias invalid");
  return Status::Ok();
}
Status ValidateIndexerCandidates(const IndexerCandidatesLaunch& x) {
  if (!x.stream || !x.tokens || x.tokens > 4096 || (x.start && x.tokens != 1) ||
      x.start >= 1048576 || x.tokens > 1048576 - x.start || x.positions != x.start + x.tokens)
    return Status::InvalidArgument("V4.1 candidate-source step invalid");
  const auto count = x.tokens * std::uint64_t(x.positions);
  if (!Valid(x.scores, count * 2, 2) || !Valid(x.output, count, 1) || !Valid(x.error_flag, 4, 4) ||
      Overlap(x.scores, x.output) || Overlap(x.scores, x.error_flag) || Overlap(x.output, x.error_flag))
    return Status::InvalidArgument("V4.1 candidate-source layout or alias invalid");
  return Status::Ok();
}
}  // namespace pih::deepseek_v41
