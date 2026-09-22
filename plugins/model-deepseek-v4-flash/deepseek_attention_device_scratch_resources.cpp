#include "pih/model/deepseek_attention_device_scratch_resources.h"

#include <limits>

#include "pih/core/checked_math.h"

namespace pih {
namespace {

Status validate(const Buffer& buffer, std::uint64_t bytes,
                std::int32_t device_ordinal) {
  if (buffer.data() == nullptr || buffer.size_bytes() != bytes ||
      buffer.generation() == 0 ||
      buffer.device().type() != DeviceType::kCuda ||
      buffer.device().index() != device_ordinal) {
    return Status::FailedPrecondition(
        "DeepSeek attention allocator returned invalid device scratch");
  }
  return Status::Ok();
}

}  // namespace

Result<DeepSeekAttentionDeviceScratchResources>
DeepSeekAttentionDeviceScratchResources::Allocate(
    Allocator& allocator, std::uint32_t maximum_queries,
    std::uint32_t maximum_ratio4_logical_pages,
    std::uint64_t context_identity, std::int32_t device_ordinal) {
  if (maximum_queries == 0 || maximum_queries > 4096 ||
      maximum_ratio4_logical_pages == 0 ||
      maximum_ratio4_logical_pages > 4096 ||
      context_identity == 0 || device_ordinal < 0) {
    return Status::InvalidArgument(
        "DeepSeek attention device scratch identity is invalid");
  }
  auto score_elements = checked_mul_u64(maximum_queries, kMaximumScoreTile);
  auto index_elements = checked_mul_u64(maximum_queries, kMaximumSparseRow);
  if (!score_elements.ok()) return score_elements.status();
  if (!index_elements.ok()) return index_elements.status();
  auto score_bytes = checked_mul_u64(*score_elements, sizeof(float));
  auto index_bytes = checked_mul_u64(*index_elements, sizeof(std::int32_t));
  auto query_elements = checked_mul_u64(maximum_queries, 64U * 128U);
  auto head_weight_elements = checked_mul_u64(maximum_queries, 64U);
  if (!score_bytes.ok()) return score_bytes.status();
  if (!index_bytes.ok()) return index_bytes.status();
  if (!query_elements.ok()) return query_elements.status();
  if (!head_weight_elements.ok()) return head_weight_elements.status();
  auto query_bytes = checked_mul_u64(*query_elements, sizeof(std::uint16_t));
  auto head_weight_bytes =
      checked_mul_u64(*head_weight_elements, sizeof(float));
  auto page_slot_bytes = checked_mul_u64(
      maximum_ratio4_logical_pages, sizeof(std::uint32_t));
  if (!query_bytes.ok()) return query_bytes.status();
  if (!head_weight_bytes.ok()) return head_weight_bytes.status();
  if (!page_slot_bytes.ok()) return page_slot_bytes.status();
  constexpr std::uint64_t error_bytes = 5 * sizeof(std::uint32_t);
  auto scores = Buffer::Allocate(allocator, *score_bytes, 256);
  if (!scores.ok()) return scores.status();
  auto indices = Buffer::Allocate(allocator, *index_bytes, 256);
  if (!indices.ok()) return indices.status();
  auto indexer_query = Buffer::Allocate(allocator, *query_bytes, 256);
  if (!indexer_query.ok()) return indexer_query.status();
  const auto qr_bytes = static_cast<std::uint64_t>(maximum_queries) * 1024U;
  const auto qr_scale_bytes = static_cast<std::uint64_t>(maximum_queries) * 8U;
  auto indexer_qr = Buffer::Allocate(allocator, qr_bytes, 256);
  if (!indexer_qr.ok()) return indexer_qr.status();
  auto indexer_qr_scales = Buffer::Allocate(allocator, qr_scale_bytes, 256);
  if (!indexer_qr_scales.ok()) return indexer_qr_scales.status();
  auto indexer_head_weight =
      Buffer::Allocate(allocator, *head_weight_bytes, 256);
  if (!indexer_head_weight.ok()) return indexer_head_weight.status();
  auto page_slots = Buffer::Allocate(allocator, *page_slot_bytes, 256);
  if (!page_slots.ok()) return page_slots.status();
  auto errors = Buffer::Allocate(allocator, error_bytes, 256);
  if (!errors.ok()) return errors.status();
  for (const auto pair : {std::pair<const Buffer*, std::uint64_t>{
                              &*scores, *score_bytes},
                          {&*indices, *index_bytes},
                          {&*indexer_query, *query_bytes},
                          {&*indexer_qr, qr_bytes},
                          {&*indexer_qr_scales, qr_scale_bytes},
                          {&*indexer_head_weight, *head_weight_bytes},
                          {&*page_slots, *page_slot_bytes},
                          {&*errors, error_bytes}}) {
    auto status = validate(*pair.first, pair.second, device_ordinal);
    if (!status.ok()) return status;
  }
  return DeepSeekAttentionDeviceScratchResources(
      std::move(*scores), std::move(*indices), std::move(*indexer_query),
      std::move(*indexer_head_weight), std::move(*page_slots),
      std::move(*errors),
      std::move(*indexer_qr), std::move(*indexer_qr_scales),
      maximum_queries, context_identity, device_ordinal);
}

std::uintptr_t DeepSeekAttentionDeviceScratchResources::error(
    std::uint32_t ordinal) const noexcept {
  return reinterpret_cast<std::uintptr_t>(errors_.data()) +
         ordinal * sizeof(std::uint32_t);
}

DeepSeekIndexSelectionArena
DeepSeekAttentionDeviceScratchResources::index_arena() const noexcept {
  return {reinterpret_cast<std::uintptr_t>(scores_.data()), index_error_u32(),
          reinterpret_cast<std::uintptr_t>(page_slots_.data()),
          static_cast<std::uint32_t>(page_slots_.size_bytes() /
                                     sizeof(std::uint32_t))};
}

std::uintptr_t
DeepSeekAttentionDeviceScratchResources::indexer_query_bf16() const noexcept {
  return reinterpret_cast<std::uintptr_t>(indexer_query_.data());
}

std::uintptr_t DeepSeekAttentionDeviceScratchResources::
    indexer_head_weight_f32() const noexcept {
  return reinterpret_cast<std::uintptr_t>(indexer_head_weight_.data());
}

std::uintptr_t
DeepSeekAttentionDeviceScratchResources::sparse_indices_i32() const noexcept {
  return reinterpret_cast<std::uintptr_t>(indices_.data());
}

std::uintptr_t
DeepSeekAttentionDeviceScratchResources::sparse_page_slots_u32() const noexcept {
  return reinterpret_cast<std::uintptr_t>(page_slots_.data());
}

std::uintptr_t
DeepSeekAttentionDeviceScratchResources::compressor_error_u32() const noexcept {
  return error(0);
}
std::uintptr_t
DeepSeekAttentionDeviceScratchResources::page_error_u32() const noexcept {
  return error(1);
}
std::uintptr_t
DeepSeekAttentionDeviceScratchResources::index_error_u32() const noexcept {
  return error(2);
}
std::uintptr_t
DeepSeekAttentionDeviceScratchResources::sparse_error_u32() const noexcept {
  return error(3);
}

std::uintptr_t DeepSeekAttentionDeviceScratchResources::
    indexer_projection_error_u32() const noexcept {
  return error(4);
}

}  // namespace pih
