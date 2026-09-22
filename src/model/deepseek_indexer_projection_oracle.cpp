#include "pih/model/deepseek_indexer_projection_oracle.h"

#include <cmath>

namespace pih {

Result<DeepSeekIndexerProjectionOracleOutput>
DeepSeekIndexerProjectionOracle::Evaluate(
    std::span<const BFloat16> qr, std::span<const BFloat16> hidden,
    std::span<const BFloat16> wq_b,
    std::span<const BFloat16> weights_proj,
    std::span<const float> frequencies,
    std::span<const std::uint32_t> positions,
    std::uint32_t table_position_count) {
  constexpr std::uint32_t kQr = 1024, kHidden = 4096;
  constexpr std::uint32_t kHeads = 64, kHeadDim = 128, kRope = 64;
  const auto tokens = positions.size();
  if (tokens == 0 || tokens > 4096 || qr.size() != tokens * kQr ||
      hidden.size() != tokens * kHidden ||
      wq_b.size() != kHeads * kHeadDim * kQr ||
      weights_proj.size() != kHeads * kHidden ||
      table_position_count == 0 ||
      frequencies.size() !=
          static_cast<std::size_t>(table_position_count) * kRope) {
    return Status::InvalidArgument(
        "DeepSeek indexer projection oracle shape is invalid");
  }
  for (const auto position : positions) {
    if (position >= table_position_count) {
      return Status::InvalidArgument(
          "DeepSeek indexer projection position is invalid");
    }
  }
  DeepSeekIndexerProjectionOracleOutput result;
  result.query.resize(tokens * kHeads * kHeadDim);
  result.head_weight.resize(tokens * kHeads);
  constexpr float kWeightScale = 0.011048543456039806F;
  for (std::size_t token = 0; token < tokens; ++token) {
    for (std::uint32_t head = 0; head < kHeads; ++head) {
      const auto output_base = (token * kHeads + head) * kHeadDim;
      for (std::uint32_t column = 0; column < kHeadDim; ++column) {
        float sum = 0.0F;
        const auto weight_base =
            (static_cast<std::size_t>(head) * kHeadDim + column) * kQr;
        for (std::uint32_t inner = 0; inner < kQr; ++inner) {
          sum += qr[token * kQr + inner].to_float() *
                 wq_b[weight_base + inner].to_float();
        }
        result.query[output_base + column] = BFloat16::FromFloat(sum);
      }
      const auto frequency_base =
          static_cast<std::size_t>(positions[token]) * kRope;
      for (std::uint32_t pair = 0; pair < kRope / 2; ++pair) {
        const auto even = output_base + 64 + pair * 2;
        const auto odd = even + 1;
        const float x = result.query[even].to_float();
        const float y = result.query[odd].to_float();
        const float cosine = frequencies[frequency_base + pair];
        const float sine = frequencies[frequency_base + pair + kRope / 2];
        result.query[even] = BFloat16::FromFloat(x * cosine - y * sine);
        result.query[odd] = BFloat16::FromFloat(y * cosine + x * sine);
      }
      float weight = 0.0F;
      const auto weight_base = static_cast<std::size_t>(head) * kHidden;
      for (std::uint32_t inner = 0; inner < kHidden; ++inner) {
        weight += hidden[token * kHidden + inner].to_float() *
                  weights_proj[weight_base + inner].to_float();
      }
      result.head_weight[token * kHeads + head] = weight * kWeightScale;
    }
  }
  return result;
}

}  // namespace pih
