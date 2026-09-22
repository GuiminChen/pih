#include "pih/model/deepseek_attention_host_staging_resources.h"

#include <limits>

#include "pih/core/checked_math.h"

namespace pih {
namespace {

Status validate(const Buffer& buffer, std::uint64_t expected_bytes) {
  if (buffer.data() == nullptr || buffer.size_bytes() != expected_bytes ||
      buffer.generation() == 0 || buffer.device() != Device::Cpu()) {
    return Status::FailedPrecondition(
        "DeepSeek attention pinned allocator returned invalid storage");
  }
  return Status::Ok();
}

}  // namespace

Result<DeepSeekAttentionHostStagingResources>
DeepSeekAttentionHostStagingResources::Allocate(
    RegisteredPinnedAllocator& allocator, std::uint32_t maximum_queries,
    std::uint32_t maximum_ratio4_logical_pages) {
  if (maximum_queries == 0 || maximum_queries > 4096) {
    return Status::InvalidArgument(
        "DeepSeek attention staging query capacity is invalid");
  }
  if (maximum_ratio4_logical_pages == 0 ||
      maximum_ratio4_logical_pages > 4096) {
    return Status::InvalidArgument(
        "DeepSeek attention page staging capacity is invalid");
  }
  auto score_elements = checked_mul_u64(maximum_queries, kMaximumScoreTile);
  if (!score_elements.ok() ||
      *score_elements > std::numeric_limits<std::uint32_t>::max()) {
    return Status::ResourceExhausted(
        "DeepSeek attention score staging capacity overflows");
  }
  auto score_bytes = checked_mul_u64(*score_elements, sizeof(float));
  if (!score_bytes.ok()) return score_bytes.status();
  auto index_elements = checked_mul_u64(maximum_queries, kMaximumSparseRow);
  if (!index_elements.ok() ||
      *index_elements > std::numeric_limits<std::uint32_t>::max()) {
    return Status::ResourceExhausted(
        "DeepSeek attention index staging capacity overflows");
  }
  auto index_bytes = checked_mul_u64(*index_elements, sizeof(std::int32_t));
  if (!index_bytes.ok()) return index_bytes.status();
  auto page_slot_bytes = checked_mul_u64(
      maximum_ratio4_logical_pages, sizeof(std::uint32_t));
  if (!page_slot_bytes.ok()) return page_slot_bytes.status();
  constexpr std::uint64_t error_bytes = 5 * sizeof(std::uint32_t);
  auto scores = Buffer::Allocate(allocator, *score_bytes, 256);
  if (!scores.ok()) return scores.status();
  auto indices = Buffer::Allocate(allocator, *index_bytes, 256);
  if (!indices.ok()) return indices.status();
  auto page_slots = Buffer::Allocate(allocator, *page_slot_bytes, 256);
  if (!page_slots.ok()) return page_slots.status();
  auto errors = Buffer::Allocate(allocator, error_bytes, 256);
  if (!errors.ok()) return errors.status();
  for (const auto pair : {std::pair<const Buffer*, std::uint64_t>{
                              &*scores, *score_bytes},
                          {&*indices, *index_bytes},
                          {&*page_slots, *page_slot_bytes},
                          {&*errors, error_bytes}}) {
    auto status = validate(*pair.first, pair.second);
    if (!status.ok()) return status;
  }
  return DeepSeekAttentionHostStagingResources(
      std::move(*scores), std::move(*indices), std::move(*page_slots),
      std::move(*errors), maximum_queries,
      maximum_ratio4_logical_pages);
}

std::uint32_t* DeepSeekAttentionHostStagingResources::error(
    std::uint32_t ordinal) noexcept {
  return static_cast<std::uint32_t*>(errors_.data()) + ordinal;
}

DeepSeekIndexSelectionHostStaging
DeepSeekAttentionHostStagingResources::index() noexcept {
  return {static_cast<float*>(scores_.data()), error(0),
          maximum_queries_ * kMaximumScoreTile,
          static_cast<std::uint32_t*>(page_slots_.data()),
          maximum_ratio4_logical_pages_};
}

DeepSeekSparseAttentionHostStaging
DeepSeekAttentionHostStagingResources::sparse() noexcept {
  return {static_cast<std::int32_t*>(indices_.data()),
          maximum_queries_ * kMaximumSparseRow, error(1),
          static_cast<std::uint32_t*>(page_slots_.data()),
          maximum_ratio4_logical_pages_};
}

std::uint32_t*
DeepSeekAttentionHostStagingResources::compressor_error() noexcept {
  return error(2);
}

std::uint32_t* DeepSeekAttentionHostStagingResources::page_error() noexcept {
  return error(3);
}

std::uint32_t*
DeepSeekAttentionHostStagingResources::indexer_projection_error() noexcept {
  return error(4);
}

}  // namespace pih
