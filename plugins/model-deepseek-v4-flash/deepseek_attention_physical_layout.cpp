#include "pih/model/deepseek_attention_physical_layout.h"

#include <limits>

#include "pih/core/canonical_hash.h"
#include "pih/core/checked_math.h"

namespace pih {
namespace {

constexpr std::uint64_t kPhysicalAlignment = 65536;
constexpr std::uint64_t kRatio4MainPageBytes = 65536;
constexpr std::uint64_t kRatio4IndexPageBytes = 16384;
constexpr std::uint64_t kRatio128PageBytes = 65536;

Result<std::uint32_t> pages_per_layer(std::uint32_t reserved_tokens,
                                      std::uint32_t ratio) {
  const auto compressed = reserved_tokens / ratio;
  const auto persistent = (compressed + 63U) / 64U;
  if (persistent == std::numeric_limits<std::uint32_t>::max()) {
    return Status::ResourceExhausted(
        "DeepSeek attention page count overflows");
  }
  return persistent + 1U;
}

Status append_extent(std::vector<DeepSeekAttentionPhysicalExtent>& extents,
                     DeepSeekAttentionPhysicalExtentKind kind,
                     std::uint64_t slot_count, std::uint64_t page_bytes,
                     std::uint64_t& allocation_bytes) {
  auto offset = checked_align_up_u64(allocation_bytes, kPhysicalAlignment);
  if (!offset.ok()) return offset.status();
  auto bytes = checked_mul_u64(slot_count, page_bytes);
  if (!bytes.ok()) return bytes.status();
  auto end = checked_add_u64(*offset, *bytes);
  if (!end.ok()) return end.status();
  extents.push_back({kind, *offset, *bytes, kPhysicalAlignment, slot_count,
                     page_bytes});
  allocation_bytes = *end;
  return Status::Ok();
}

}  // namespace

Result<DeepSeekAttentionPhysicalLayout>
DeepSeekAttentionPhysicalLayout::Compile(
    DeepSeekStagePlan stage, std::uint32_t maximum_sequences,
    std::uint32_t reserved_tokens_per_sequence) {
  if (stage.layers.first_layer > stage.layers.last_layer ||
      stage.layers.last_layer >= 43 || maximum_sequences == 0 ||
      reserved_tokens_per_sequence == 0 ||
      reserved_tokens_per_sequence > 1048576) {
    return Status::InvalidArgument(
        "DeepSeek attention physical layout capacity is invalid");
  }
  std::vector<std::uint32_t> layers;
  layers.reserve(stage.layers.last_layer - stage.layers.first_layer + 1U);
  for (std::uint32_t layer = stage.layers.first_layer;
       layer <= stage.layers.last_layer; ++layer) {
    layers.push_back(layer);
  }
  auto fixed = DeepSeekFixedStateLayout::Build(layers, stage.owns_dspark);
  if (!fixed.ok()) return fixed.status();
  std::uint32_t ratio4_layers = 0;
  std::uint32_t ratio128_layers = 0;
  for (const auto& descriptor : fixed->descriptors()) {
    ratio4_layers += descriptor.kind == DeepSeekFixedLayerKind::kRatio4;
    ratio128_layers += descriptor.kind == DeepSeekFixedLayerKind::kRatio128;
  }
  auto pages4 = pages_per_layer(reserved_tokens_per_sequence, 4);
  auto pages128 = pages_per_layer(reserved_tokens_per_sequence, 128);
  if (!pages4.ok()) return pages4.status();
  if (!pages128.ok()) return pages128.status();
  auto per_sequence4 = checked_mul_u64(ratio4_layers, *pages4);
  auto per_sequence128 = checked_mul_u64(ratio128_layers, *pages128);
  if (!per_sequence4.ok()) return per_sequence4.status();
  if (!per_sequence128.ok()) return per_sequence128.status();
  if (*per_sequence4 == 0 || *per_sequence128 == 0 ||
      *per_sequence4 > UINT32_MAX || *per_sequence128 > UINT32_MAX) {
    return Status::InvalidArgument(
        "DeepSeek stage must own both compressed attention kinds");
  }
  auto total4 = checked_mul_u64(maximum_sequences, *per_sequence4);
  auto total128 = checked_mul_u64(maximum_sequences, *per_sequence128);
  if (!total4.ok()) return total4.status();
  if (!total128.ok()) return total128.status();
  if (*total4 > UINT32_MAX || *total128 > UINT32_MAX) {
    return Status::ResourceExhausted(
        "DeepSeek attention physical layout slots overflow");
  }

  DeepSeekAttentionPhysicalLayout result;
  result.fixed_layout_ = std::move(*fixed);
  result.ratio4_page_pairs_per_sequence_ =
      static_cast<std::uint32_t>(*per_sequence4);
  result.ratio128_pages_per_sequence_ =
      static_cast<std::uint32_t>(*per_sequence128);
  result.ratio4_pair_count_ = static_cast<std::uint32_t>(*total4);
  result.ratio128_page_count_ = static_cast<std::uint32_t>(*total128);
  Status status = append_extent(
      result.extents_, DeepSeekAttentionPhysicalExtentKind::kRatio4Main,
      *total4, kRatio4MainPageBytes, result.allocation_bytes_);
  if (status.ok()) {
    status = append_extent(
        result.extents_, DeepSeekAttentionPhysicalExtentKind::kRatio4Index,
        *total4, kRatio4IndexPageBytes, result.allocation_bytes_);
  }
  if (status.ok()) {
    status = append_extent(
        result.extents_, DeepSeekAttentionPhysicalExtentKind::kRatio128Main,
        *total128, kRatio128PageBytes, result.allocation_bytes_);
  }
  if (!status.ok()) return status;

  auto hash = CanonicalHashBuilder::Create(
      "pih:deepseek-attention-physical-layout:v1", 29);
  if (!hash.ok()) return hash.status();
  status = hash->add_u32(1, stage.rank);
  if (status.ok()) status = hash->add_u32(2, stage.layers.first_layer);
  if (status.ok()) status = hash->add_u32(3, stage.layers.last_layer);
  if (status.ok()) status = hash->add_u32(4, stage.owns_embedding);
  if (status.ok()) status = hash->add_u32(5, stage.owns_lm_head);
  if (status.ok()) status = hash->add_u32(6, stage.owns_dspark);
  if (status.ok()) status = hash->add_u32(7, maximum_sequences);
  if (status.ok()) status = hash->add_u32(8, reserved_tokens_per_sequence);
  if (status.ok()) status = hash->add_u32(9, result.version());
  if (status.ok()) {
    status = hash->add_u32(10, result.ratio4_page_pairs_per_sequence_);
  }
  if (status.ok()) {
    status = hash->add_u32(11, result.ratio128_pages_per_sequence_);
  }
  if (status.ok()) status = hash->add_u32(12, result.ratio4_pair_count_);
  if (status.ok()) status = hash->add_u32(13, result.ratio128_page_count_);
  if (status.ok()) status = hash->add_u64(14, result.allocation_bytes_);
  std::uint16_t field = 15;
  for (const auto& extent : result.extents_) {
    if (status.ok()) {
      status = hash->add_u32(field++, static_cast<std::uint32_t>(extent.kind));
    }
    if (status.ok()) status = hash->add_u64(field++, extent.offset_bytes);
    if (status.ok()) status = hash->add_u64(field++, extent.bytes);
    if (status.ok()) status = hash->add_u64(field++, extent.slot_count);
    if (status.ok()) status = hash->add_u64(field++, extent.page_bytes);
  }
  if (!status.ok()) return status;
  auto identity = hash->finalize();
  if (!identity.ok()) return identity.status();
  result.identity_ = *identity;
  return result;
}

}  // namespace pih
