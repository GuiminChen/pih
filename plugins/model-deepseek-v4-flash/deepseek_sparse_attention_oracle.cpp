#include "pih/model/deepseek_sparse_attention_oracle.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace pih {

Result<std::vector<BFloat16>> DeepSeekSparseAttentionOracle::Evaluate(
    std::span<const BFloat16> query,
    std::span<const BFloat16> latent_kv,
    std::span<const float> attention_sink,
    std::span<const std::int32_t> indices,
    std::uint32_t head_count, std::uint32_t kv_count,
    float softmax_scale) {
  if (head_count == 0 || head_count > kMaximumHeads || kv_count == 0 ||
      indices.empty() || indices.size() > kMaximumIndices ||
      query.size() != static_cast<std::size_t>(head_count) * kHeadDim ||
      latent_kv.size() != static_cast<std::size_t>(kv_count) * kHeadDim ||
      attention_sink.size() != head_count || !std::isfinite(softmax_scale) ||
      softmax_scale <= 0.0F) {
    return Status::InvalidArgument(
        "DeepSeek sparse attention oracle shape is invalid");
  }
  bool has_valid_index = false;
  for (const auto index : indices) {
    if (index < -1 || index >= static_cast<std::int32_t>(kv_count)) {
      return Status::InvalidArgument(
          "DeepSeek sparse attention index is invalid");
    }
    has_valid_index |= index >= 0;
  }
  if (!has_valid_index) {
    return Status::InvalidArgument(
        "DeepSeek sparse attention has no visible KV position");
  }
  for (const auto sink : attention_sink) {
    if (!std::isfinite(sink)) {
      return Status::InvalidArgument(
          "DeepSeek sparse attention sink is nonfinite");
    }
  }

  std::vector<BFloat16> output(
      static_cast<std::size_t>(head_count) * kHeadDim);
  std::vector<float> accumulator(kHeadDim);
  std::vector<float> scores(kTileIndices);
  for (std::uint32_t head = 0; head < head_count; ++head) {
    std::fill(accumulator.begin(), accumulator.end(), 0.0F);
    float running_max = -std::numeric_limits<float>::infinity();
    float running_sum = 0.0F;
    for (std::size_t tile = 0; tile < indices.size();
         tile += kTileIndices) {
      const auto count = std::min<std::size_t>(kTileIndices,
                                               indices.size() - tile);
      float tile_max = -std::numeric_limits<float>::infinity();
      for (std::size_t item = 0; item < count; ++item) {
        const auto index = indices[tile + item];
        if (index < 0) {
          scores[item] = -std::numeric_limits<float>::infinity();
          continue;
        }
        float score = 0.0F;
        for (std::uint32_t column = 0; column < kHeadDim; ++column) {
          score += query[static_cast<std::size_t>(head) * kHeadDim + column]
                       .to_float() *
                   latent_kv[static_cast<std::size_t>(index) * kHeadDim +
                             column]
                       .to_float();
        }
        score *= softmax_scale;
        if (!std::isfinite(score)) {
          return Status::InvalidArgument(
              "DeepSeek sparse attention score is nonfinite");
        }
        scores[item] = score;
        tile_max = std::max(tile_max, score);
      }
      const auto next_max = std::max(running_max, tile_max);
      const auto previous_scale = std::isinf(running_max)
                                      ? 0.0F
                                      : std::exp(running_max - next_max);
      running_sum *= previous_scale;
      for (auto& value : accumulator) value *= previous_scale;
      for (std::size_t item = 0; item < count; ++item) {
        if (indices[tile + item] < 0) continue;
        const auto weight = std::exp(scores[item] - next_max);
        running_sum += weight;
        const auto bf16_weight = BFloat16::FromFloat(weight).to_float();
        const auto index = static_cast<std::size_t>(indices[tile + item]);
        for (std::uint32_t column = 0; column < kHeadDim; ++column) {
          accumulator[column] +=
              bf16_weight *
              latent_kv[index * kHeadDim + column].to_float();
        }
      }
      running_max = next_max;
    }
    const auto denominator =
        running_sum + std::exp(attention_sink[head] - running_max);
    if (!std::isfinite(denominator) || denominator <= 0.0F) {
      return Status::InvalidArgument(
          "DeepSeek sparse attention denominator is invalid");
    }
    for (std::uint32_t column = 0; column < kHeadDim; ++column) {
      const auto value = accumulator[column] / denominator;
      if (!std::isfinite(value)) {
        return Status::InvalidArgument(
            "DeepSeek sparse attention output is nonfinite");
      }
      output[static_cast<std::size_t>(head) * kHeadDim + column] =
          BFloat16::FromFloat(value);
    }
  }
  return output;
}

Result<std::vector<BFloat16>> DeepSeekSparseAttentionOracle::EvaluatePaged(
    std::span<const BFloat16> query,
    std::span<const BFloat16> recent_kv,
    std::span<const BFloat16> compressed_kv,
    std::span<const std::uint32_t> page_slots,
    std::span<const float> attention_sink,
    std::span<const std::int32_t> indices,
    std::uint32_t head_count, std::uint32_t kv_count,
    std::int32_t recent_physical_offset,
    std::int32_t compressed_physical_offset,
    std::uint32_t compressed_slot_count,
    std::uint32_t physical_page_count, float softmax_scale) {
  const auto logical_pages = (compressed_slot_count + 63U) / 64U;
  if (recent_physical_offset < 0 || compressed_physical_offset < 0 ||
      compressed_slot_count == 0 || physical_page_count == 0 ||
      page_slots.size() != logical_pages ||
      recent_kv.size() != 128ULL * kHeadDim ||
      compressed_kv.size() !=
          static_cast<std::size_t>(physical_page_count) * 64U * kHeadDim ||
      static_cast<std::uint64_t>(recent_physical_offset) + 128U > kv_count ||
      static_cast<std::uint64_t>(compressed_physical_offset) +
              compressed_slot_count >
          kv_count) {
    return Status::InvalidArgument(
        "DeepSeek paged sparse attention oracle shape is invalid");
  }
  const auto recent_begin = static_cast<std::uint32_t>(recent_physical_offset);
  const auto compressed_begin =
      static_cast<std::uint32_t>(compressed_physical_offset);
  if (!(recent_begin + 128U <= compressed_begin ||
        compressed_begin + compressed_slot_count <= recent_begin)) {
    return Status::InvalidArgument(
        "DeepSeek paged sparse attention ranges overlap");
  }
  std::vector<BFloat16> logical(
      static_cast<std::size_t>(kv_count) * kHeadDim);
  std::copy(recent_kv.begin(), recent_kv.end(),
            logical.begin() + static_cast<std::size_t>(recent_begin) * kHeadDim);
  for (std::uint32_t logical_slot = 0;
       logical_slot < compressed_slot_count; ++logical_slot) {
    const auto physical_page = page_slots[logical_slot / 64U];
    if (physical_page >= physical_page_count) {
      return Status::InvalidArgument(
          "DeepSeek paged sparse attention page slot is invalid");
    }
    const auto physical_slot = physical_page * 64U + logical_slot % 64U;
    std::copy_n(
        compressed_kv.begin() +
            static_cast<std::size_t>(physical_slot) * kHeadDim,
        kHeadDim,
        logical.begin() +
            static_cast<std::size_t>(compressed_begin + logical_slot) *
                kHeadDim);
  }
  return Evaluate(query, logical, attention_sink, indices, head_count,
                  kv_count, softmax_scale);
}

}  // namespace pih
