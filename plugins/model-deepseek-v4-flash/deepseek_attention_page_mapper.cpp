#include "pih/model/deepseek_attention_page_mapper.h"

#include <limits>

namespace pih {
namespace {

bool compressed(const DeepSeekStatePoolGeometry& geometry) {
  return geometry.pool_kind == DeepSeekStatePoolKind::kRatio4MainBf16 ||
         geometry.pool_kind == DeepSeekStatePoolKind::kRatio4IndexBf16 ||
         geometry.pool_kind == DeepSeekStatePoolKind::kRatio128MainBf16;
}

}  // namespace

Result<DeepSeekCompressedCoverage> DeepSeekAttentionPageMapper::Coverage(
    std::uint64_t source_tokens,
    const DeepSeekStatePoolGeometry& geometry) {
  auto status = geometry.validate();
  if (!status.ok()) return status;
  if (!compressed(geometry)) {
    return Status::InvalidArgument(
        "DeepSeek compressed coverage requires a compressed pool");
  }
  const auto ratio = static_cast<std::uint64_t>(geometry.logical_ratio);
  const auto units =
      static_cast<std::uint64_t>(geometry.storage_units_per_page);
  const auto slots = source_tokens / ratio;
  if (slots > std::numeric_limits<std::uint64_t>::max() - (units - 1U)) {
    return Status::ResourceExhausted(
        "DeepSeek compressed page count overflows");
  }
  return DeepSeekCompressedCoverage{
      source_tokens, slots, (slots + units - 1U) / units,
      static_cast<std::uint32_t>(source_tokens % ratio)};
}

Result<DeepSeekCompressedPageLocation>
DeepSeekAttentionPageMapper::LocateStoredSlot(
    std::uint64_t slot_ordinal,
    const DeepSeekStatePoolGeometry& geometry) {
  auto status = geometry.validate();
  if (!status.ok()) return status;
  if (!compressed(geometry)) {
    return Status::InvalidArgument(
        "DeepSeek stored-slot mapping requires a compressed pool");
  }
  const auto units =
      static_cast<std::uint64_t>(geometry.storage_units_per_page);
  return DeepSeekCompressedPageLocation{
      slot_ordinal, slot_ordinal / units,
      static_cast<std::uint32_t>(slot_ordinal % units)};
}

}  // namespace pih
