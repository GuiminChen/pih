#include "indexer_input.h"
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
Status ValidateIndexerQuantize(const IndexerQuantizeLaunch& x) {
  if (!x.stream || !x.tokens || x.tokens > 4096 ||
      (x.heads != 1 && x.heads != 4 && x.heads != 8 && x.heads != 16 && x.heads != 32) ||
      !Valid(x.values, x.tokens * std::uint64_t(x.heads) * 128 * 2, 2) ||
      !Valid(x.error_flag, 4, 4) || Overlap(x.values, x.error_flag))
    return Status::InvalidArgument("V4.1 indexer quantization geometry or buffer invalid");
  return Status::Ok();
}
Status ValidateIndexerQuery(const IndexerQueryLaunch& x) {
  const auto projection = ValidateFp8Linear(x.projection); if (!projection.ok()) return projection;
  const auto rope = ValidateRopeApply(x.rope); if (!rope.ok()) return rope;
  const auto quantize = ValidateIndexerQuantize(x.quantize); if (!quantize.ok()) return quantize;
  const auto& p = x.projection;
  const auto& q = x.quantize;
  if (q.heads == 1 || p.in_features != 1280 || p.out_features != q.heads * 128 ||
      x.rope.width != 128 || x.rope.heads != q.heads || x.rope.inverse ||
      p.rows != q.tokens || x.rope.tokens != q.tokens || p.stream != q.stream || x.rope.stream != q.stream ||
      !Same(p.output, x.rope.input) || !Same(x.rope.output, q.values) ||
      !Same(p.error_flag, q.error_flag) || !Same(x.rope.error_flag, q.error_flag))
    return Status::InvalidArgument("V4.1 indexer query stage connection invalid");
  const std::array reads{p.input, p.weight, p.weight_scales, x.rope.phases};
  const std::array writes{p.quantized, p.activation_scales, p.output, q.values, p.error_flag};
  for (std::size_t i = 0; i < writes.size(); ++i) {
    for (const auto read : reads)
      if (Overlap(read, writes[i])) return Status::InvalidArgument("Indexer query overwrites a live input");
    for (std::size_t j = 0; j < i; ++j)
      if (Overlap(writes[i], writes[j]) && !(i == 3 && j == 2 && Same(writes[i], writes[j])))
        return Status::InvalidArgument("Indexer query writable alias");
  }
  return Status::Ok();
}
Status LaunchIndexerQuery(const IndexerQueryLaunch& x) {
  const auto validation = ValidateIndexerQuery(x); if (!validation.ok()) return validation;
  const auto projection = LaunchFp8Linear(x.projection); if (!projection.ok()) return projection;
  const auto rope = LaunchRopeApply(x.rope); if (!rope.ok()) return rope;
  return LaunchIndexerQuantize(x.quantize);
}
Status ValidateIndexerKeyProjection(const IndexerKeyProjectionLaunch& x) {
  if (!x.stream || !x.rows || x.rows > 4096 || !Valid(x.input, x.rows * 512ULL * 2, 2) ||
      !Valid(x.weight, 128 * 512 * 2, 2) || !Valid(x.output, x.rows * 128ULL * 2, 2) ||
      !Valid(x.error_flag, 4, 4) || Overlap(x.output, x.error_flag))
    return Status::InvalidArgument("V4.1 index-key projection geometry or buffer invalid");
  for (const auto read : std::array{x.input, x.weight})
    for (const auto write : std::array{x.output, x.error_flag})
      if (Overlap(read, write)) return Status::InvalidArgument("Index-key projection overwrites input");
  return Status::Ok();
}
Status ValidateIndexerKeyCache(const IndexerKeyCacheLaunch& x) {
  const auto quantize = ValidateIndexerQuantize(x.quantize); if (!quantize.ok()) return quantize;
  if (x.quantize.heads != 1 || !x.capacity || x.capacity > 1048576 || x.first_slot >= x.capacity ||
      x.quantize.tokens > x.capacity - x.first_slot || !Valid(x.cache, x.capacity * 128ULL * 2, 2) ||
      Overlap(x.cache, x.quantize.values) || Overlap(x.cache, x.quantize.error_flag))
    return Status::InvalidArgument("V4.1 index-key cache extent or alias invalid");
  return Status::Ok();
}
Status ValidateIndexerKey(const IndexerKeyLaunch& x) {
  const auto projection = ValidateIndexerKeyProjection(x.projection); if (!projection.ok()) return projection;
  const auto norm = ValidateRmsNorm(x.norm); if (!norm.ok()) return norm;
  const auto rope = ValidateRopeApply(x.rope); if (!rope.ok()) return rope;
  const auto cache = ValidateIndexerKeyCache(x.cache); if (!cache.ok()) return cache;
  const auto& p = x.projection;
  const auto& q = x.cache.quantize;
  if (x.norm.width != 128 || x.norm.rows != p.rows || x.rope.width != 128 || x.rope.heads != 1 ||
      x.rope.inverse || x.rope.tokens != p.rows || q.tokens != p.rows ||
      x.norm.stream != p.stream || x.rope.stream != p.stream || q.stream != p.stream ||
      !Same(p.output, x.norm.input) || !Same(x.norm.output, x.rope.input) || !Same(x.rope.output, q.values) ||
      !Same(p.error_flag, x.norm.error_flag) || !Same(p.error_flag, x.rope.error_flag) || !Same(p.error_flag, q.error_flag))
    return Status::InvalidArgument("V4.1 index-key stage connection invalid");
  const std::array reads{p.input, p.weight, x.norm.weight, x.rope.phases};
  const std::array writes{p.output, x.norm.output, q.values, x.cache.cache, p.error_flag};
  for (std::size_t i = 0; i < writes.size(); ++i) {
    for (const auto read : reads)
      if (Overlap(read, writes[i])) return Status::InvalidArgument("Index-key chain overwrites a live input");
    for (std::size_t j = 0; j < i; ++j)
      if (Overlap(writes[i], writes[j]) && !(i == 2 && j == 1 && Same(writes[i], writes[j])))
        return Status::InvalidArgument("Index-key chain writable alias");
  }
  return Status::Ok();
}
Status LaunchIndexerKey(const IndexerKeyLaunch& x) {
  const auto validation = ValidateIndexerKey(x); if (!validation.ok()) return validation;
  const auto projection = LaunchIndexerKeyProjection(x.projection); if (!projection.ok()) return projection;
  const auto norm = LaunchRmsNorm(x.norm); if (!norm.ok()) return norm;
  const auto rope = LaunchRopeApply(x.rope); if (!rope.ok()) return rope;
  return LaunchIndexerKeyCache(x.cache);
}
}  // namespace pih::deepseek_v41
