#include "pih/model/deepseek_attention_pool_geometry.h"

#include <limits>

namespace pih {
namespace {

template <class T>
void put(std::span<std::byte> out, std::size_t offset, T value) {
  for (std::size_t index = 0; index < sizeof(T); ++index) {
    out[offset + index] =
        static_cast<std::byte>((value >> (index * 8U)) & 0xFFU);
  }
}

template <class T>
T get(std::span<const std::byte> in, std::size_t offset) {
  T value = 0;
  for (std::size_t index = 0; index < sizeof(T); ++index) {
    value |= static_cast<T>(std::to_integer<std::uint8_t>(in[offset + index]))
             << (index * 8U);
  }
  return value;
}

bool known_kind(DeepSeekStatePoolKind kind) {
  return kind >= DeepSeekStatePoolKind::kRecentBf16 &&
         kind <= DeepSeekStatePoolKind::kRatio128MainBf16;
}

}  // namespace

Status DeepSeekStatePoolGeometry::validate() const {
  constexpr auto kKnownLayers =
      kSwaOnlyLayer | kC4aLayer | kC128aLayer | kDsparkLayer;
  if (!known_kind(pool_kind) || layer_type_mask == 0 ||
      (layer_type_mask & ~kKnownLayers) != 0 || logical_ratio == 0 ||
      storage_units_per_page == 0 || bytes_per_storage_unit == 0 ||
      page_alignment_bytes == 0 ||
      (page_alignment_bytes & (page_alignment_bytes - 1U)) != 0 ||
      handle_bytes != 8 || handles_per_logical_page == 0 || reserved != 0) {
    return Status::InvalidArgument(
        "DeepSeek attention pool geometry fields are invalid");
  }
  const auto coverage = static_cast<std::uint64_t>(logical_ratio) *
                        storage_units_per_page;
  const auto raw = static_cast<std::uint64_t>(storage_units_per_page) *
                   bytes_per_storage_unit;
  if (coverage > std::numeric_limits<std::uint32_t>::max() ||
      logical_coverage_tokens != coverage || raw_page_bytes != raw ||
      raw > std::numeric_limits<std::uint64_t>::max() -
                (page_alignment_bytes - 1U)) {
    return Status::InvalidArgument(
        "DeepSeek attention pool geometry arithmetic is invalid");
  }
  const auto alignment = static_cast<std::uint64_t>(page_alignment_bytes);
  const auto physical = (raw + alignment - 1U) & ~(alignment - 1U);
  if (physical_page_bytes != physical) {
    return Status::InvalidArgument(
        "DeepSeek attention pool physical page bytes are invalid");
  }
  const bool ratio4 = pool_kind == DeepSeekStatePoolKind::kRatio4MainBf16 ||
                      pool_kind == DeepSeekStatePoolKind::kRatio4IndexBf16;
  if ((ratio4 && (paired_commit_group != 1 ||
                  handles_per_logical_page != 2)) ||
      (!ratio4 && (paired_commit_group != 0 ||
                   handles_per_logical_page != 1))) {
    return Status::InvalidArgument(
        "DeepSeek attention pool commit geometry is invalid");
  }
  const auto canonical = CanonicalBf16();
  const auto& expected =
      canonical[static_cast<std::size_t>(pool_kind) - 1U];
  if (layer_type_mask != expected.layer_type_mask ||
      logical_ratio != expected.logical_ratio ||
      logical_coverage_tokens != expected.logical_coverage_tokens ||
      storage_units_per_page != expected.storage_units_per_page ||
      bytes_per_storage_unit != expected.bytes_per_storage_unit ||
      raw_page_bytes != expected.raw_page_bytes ||
      page_alignment_bytes != expected.page_alignment_bytes ||
      physical_page_bytes != expected.physical_page_bytes ||
      handle_bytes != expected.handle_bytes ||
      handles_per_logical_page != expected.handles_per_logical_page ||
      paired_commit_group != expected.paired_commit_group) {
    return Status::InvalidArgument(
        "DeepSeek BF16 attention pool geometry is not canonical");
  }
  return Status::Ok();
}

Result<std::array<std::byte, DeepSeekStatePoolGeometry::kWireBytes>>
DeepSeekStatePoolGeometry::encode() const {
  auto status = validate();
  if (!status.ok()) return status;
  std::array<std::byte, kWireBytes> wire{};
  auto out = std::span(wire);
  put(out, 0, static_cast<std::uint16_t>(pool_kind));
  put(out, 2, layer_type_mask);
  put(out, 4, logical_ratio);
  put(out, 8, logical_coverage_tokens);
  put(out, 12, storage_units_per_page);
  put(out, 16, bytes_per_storage_unit);
  put(out, 20, raw_page_bytes);
  put(out, 28, page_alignment_bytes);
  put(out, 32, physical_page_bytes);
  put(out, 40, handle_bytes);
  put(out, 42, handles_per_logical_page);
  put(out, 44, paired_commit_group);
  put(out, 46, reserved);
  return wire;
}

Result<DeepSeekStatePoolGeometry> DeepSeekStatePoolGeometry::Decode(
    std::span<const std::byte> wire) {
  if (wire.size() != kWireBytes) {
    return Status::InvalidArgument(
        "DeepSeek attention pool geometry wire size is invalid");
  }
  DeepSeekStatePoolGeometry value;
  value.pool_kind = static_cast<DeepSeekStatePoolKind>(get<std::uint16_t>(wire, 0));
  value.layer_type_mask = get<std::uint16_t>(wire, 2);
  value.logical_ratio = get<std::uint32_t>(wire, 4);
  value.logical_coverage_tokens = get<std::uint32_t>(wire, 8);
  value.storage_units_per_page = get<std::uint32_t>(wire, 12);
  value.bytes_per_storage_unit = get<std::uint32_t>(wire, 16);
  value.raw_page_bytes = get<std::uint64_t>(wire, 20);
  value.page_alignment_bytes = get<std::uint32_t>(wire, 28);
  value.physical_page_bytes = get<std::uint64_t>(wire, 32);
  value.handle_bytes = get<std::uint16_t>(wire, 40);
  value.handles_per_logical_page = get<std::uint16_t>(wire, 42);
  value.paired_commit_group = get<std::uint16_t>(wire, 44);
  value.reserved = get<std::uint16_t>(wire, 46);
  auto status = value.validate();
  if (!status.ok()) return status;
  return value;
}

std::array<DeepSeekStatePoolGeometry, 4>
DeepSeekStatePoolGeometry::CanonicalBf16() {
  return {{{DeepSeekStatePoolKind::kRecentBf16,
            kSwaOnlyLayer | kC4aLayer | kC128aLayer | kDsparkLayer,
            1, 128, 128, 1024, 131072, 256, 131072, 8, 1, 0, 0},
           {DeepSeekStatePoolKind::kRatio4MainBf16, kC4aLayer,
            4, 256, 64, 1024, 65536, 256, 65536, 8, 2, 1, 0},
           {DeepSeekStatePoolKind::kRatio4IndexBf16, kC4aLayer,
            4, 256, 64, 256, 16384, 256, 16384, 8, 2, 1, 0},
           {DeepSeekStatePoolKind::kRatio128MainBf16, kC128aLayer,
            128, 8192, 64, 1024, 65536, 256, 65536, 8, 1, 0, 0}}};
}

}  // namespace pih
