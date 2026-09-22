#include "pih/model/deepseek_index_score_oracle.h"

#include <algorithm>
#include <cmath>

namespace pih {

Result<std::vector<float>> DeepSeekIndexScoreOracle::Evaluate(
    std::span<const BFloat16> query,
    std::span<const BFloat16> index_kv,
    std::span<const float> head_weight,
    std::uint32_t query_count, std::uint32_t head_count,
    std::uint32_t slot_count) {
  const auto query_elements = static_cast<std::uint64_t>(query_count) *
                              head_count * kHeadDim;
  const auto kv_elements = static_cast<std::uint64_t>(slot_count) * kHeadDim;
  const auto weight_elements = static_cast<std::uint64_t>(query_count) *
                               head_count;
  if (query_count == 0 || query_count > 4096 || head_count == 0 ||
      head_count > kMaximumHeads || slot_count == 0 ||
      slot_count > kMaximumSlotTile || query.size() != query_elements ||
      index_kv.size() != kv_elements ||
      head_weight.size() != weight_elements) {
    return Status::InvalidArgument("DeepSeek index score oracle shape is invalid");
  }
  std::vector<float> output(static_cast<std::size_t>(query_count) * slot_count);
  for (std::uint32_t query_ordinal = 0; query_ordinal < query_count;
       ++query_ordinal) {
    for (std::uint32_t slot = 0; slot < slot_count; ++slot) {
      float reduced = 0.0F;
      for (std::uint32_t head = 0; head < head_count; ++head) {
        float dot = 0.0F;
        const auto query_base =
            (static_cast<std::uint64_t>(query_ordinal) * head_count + head) *
            kHeadDim;
        const auto kv_base = static_cast<std::uint64_t>(slot) * kHeadDim;
        for (std::uint32_t column = 0; column < kHeadDim; ++column) {
          dot += query[query_base + column].to_float() *
                 index_kv[kv_base + column].to_float();
        }
        reduced += std::max(dot, 0.0F) *
                   head_weight[static_cast<std::uint64_t>(query_ordinal) *
                                   head_count +
                               head];
      }
      if (!std::isfinite(reduced)) {
        return Status::InvalidArgument(
            "DeepSeek index score oracle result is nonfinite");
      }
      output[static_cast<std::size_t>(query_ordinal) * slot_count + slot] =
          reduced;
    }
  }
  return output;
}

Result<std::vector<float>> DeepSeekIndexScoreOracle::EvaluatePaged(
    std::span<const BFloat16> query,
    std::span<const BFloat16> physical_index_kv,
    std::span<const float> head_weight,
    std::span<const std::uint32_t> page_slots,
    std::uint32_t query_count, std::uint32_t head_count,
    std::uint32_t slot_base, std::uint32_t slot_count,
    std::uint32_t physical_page_count) {
  const auto logical_end = static_cast<std::uint64_t>(slot_base) + slot_count;
  const auto query_elements = static_cast<std::uint64_t>(query_count) *
                              head_count * kHeadDim;
  const auto weight_elements = static_cast<std::uint64_t>(query_count) *
                               head_count;
  const auto physical_elements =
      static_cast<std::uint64_t>(physical_page_count) * 64U * kHeadDim;
  if (query_count == 0 || query_count > 4096 || head_count == 0 ||
      head_count > kMaximumHeads || slot_count == 0 ||
      slot_count > kMaximumSlotTile || physical_page_count == 0 ||
      logical_end > static_cast<std::uint64_t>(page_slots.size()) * 64U ||
      query.size() != query_elements ||
      head_weight.size() != weight_elements ||
      physical_index_kv.size() != physical_elements) {
    return Status::InvalidArgument(
        "DeepSeek paged index score oracle shape is invalid");
  }
  std::vector<float> output(static_cast<std::size_t>(query_count) * slot_count);
  for (std::uint32_t q = 0; q < query_count; ++q) {
    for (std::uint32_t slot = 0; slot < slot_count; ++slot) {
      const auto logical_slot = slot_base + slot;
      const auto physical_page = page_slots[logical_slot / 64U];
      if (physical_page >= physical_page_count) {
        return Status::InvalidArgument(
            "DeepSeek paged index score oracle page is invalid");
      }
      const auto physical_slot =
          static_cast<std::uint64_t>(physical_page) * 64U +
          logical_slot % 64U;
      float reduced = 0.0F;
      for (std::uint32_t head = 0; head < head_count; ++head) {
        float dot = 0.0F;
        const auto query_base =
            (static_cast<std::uint64_t>(q) * head_count + head) * kHeadDim;
        const auto kv_base = physical_slot * kHeadDim;
        for (std::uint32_t column = 0; column < kHeadDim; ++column) {
          dot += query[query_base + column].to_float() *
                 physical_index_kv[kv_base + column].to_float();
        }
        reduced += std::max(dot, 0.0F) *
                   head_weight[static_cast<std::uint64_t>(q) * head_count +
                               head];
      }
      if (!std::isfinite(reduced)) {
        return Status::InvalidArgument(
            "DeepSeek paged index score oracle result is nonfinite");
      }
      output[static_cast<std::size_t>(q) * slot_count + slot] = reduced;
    }
  }
  return output;
}

}  // namespace pih
