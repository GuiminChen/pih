#include "pih/model/deepseek_compressor_pooling_oracle.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace pih {

Result<DeepSeekCompressorPoolingOracle>
DeepSeekCompressorPoolingOracle::Create(std::uint32_t ratio,
                                         std::uint32_t head_dim) {
  if ((ratio != 4 && ratio != 128) || head_dim == 0 || head_dim > 512) {
    return Status::InvalidArgument(
        "DeepSeek compressor pooling shape is invalid");
  }
  DeepSeekCompressorPoolingOracle value;
  value.ratio_ = ratio;
  value.head_dim_ = head_dim;
  value.projection_dim_ = (ratio == 4 ? 2U : 1U) * head_dim;
  const auto state_rows = (ratio == 4 ? 2U : 1U) * ratio;
  value.kv_state_.assign(static_cast<std::size_t>(state_rows) *
                             value.projection_dim_,
                         0.0F);
  value.score_state_.assign(
      static_cast<std::size_t>(state_rows) * value.projection_dim_,
      -std::numeric_limits<float>::infinity());
  return value;
}

Result<std::vector<float>> DeepSeekCompressorPoolingOracle::step(
    std::uint32_t absolute_position, std::span<const float> kv_projection,
    std::span<const float> gate_projection, std::span<const float> ape_row) {
  if (absolute_position >= 1048576U ||
      absolute_position != completed_slots_ * ratio_ + remainder_tokens_ ||
      kv_projection.size() != projection_dim_ ||
      gate_projection.size() != projection_dim_ ||
      ape_row.size() != projection_dim_) {
    return Status::InvalidArgument(
        "DeepSeek compressor pooling step is not contiguous");
  }
  const auto row = (ratio_ == 4 ? ratio_ : 0U) + remainder_tokens_;
  const auto base = static_cast<std::size_t>(row) * projection_dim_;
  for (std::uint32_t column = 0; column < projection_dim_; ++column) {
    if (!std::isfinite(kv_projection[column]) ||
        !std::isfinite(gate_projection[column]) ||
        !std::isfinite(ape_row[column])) {
      return Status::InvalidArgument(
          "DeepSeek compressor pooling input is nonfinite");
    }
    if (!std::isfinite(gate_projection[column] + ape_row[column])) {
      return Status::InvalidArgument(
          "DeepSeek compressor pooling score is nonfinite");
    }
  }
  for (std::uint32_t column = 0; column < projection_dim_; ++column) {
    kv_state_[base + column] = kv_projection[column];
    score_state_[base + column] = gate_projection[column] + ape_row[column];
  }
  ++remainder_tokens_;
  if (remainder_tokens_ != ratio_) return std::vector<float>{};

  const auto candidate_count = ratio_ == 4 ? 2U * ratio_ : ratio_;
  std::vector<float> output(head_dim_, 0.0F);
  for (std::uint32_t column = 0; column < head_dim_; ++column) {
    float maximum = -std::numeric_limits<float>::infinity();
    for (std::uint32_t candidate = 0; candidate < candidate_count;
         ++candidate) {
      const auto source_column =
          ratio_ == 4 && candidate >= ratio_ ? column + head_dim_ : column;
      maximum = std::max(
          maximum,
          score_state_[static_cast<std::size_t>(candidate) * projection_dim_ +
                       source_column]);
    }
    float denominator = 0.0F;
    for (std::uint32_t candidate = 0; candidate < candidate_count;
         ++candidate) {
      const auto source_column =
          ratio_ == 4 && candidate >= ratio_ ? column + head_dim_ : column;
      const auto weight = std::exp(
          score_state_[static_cast<std::size_t>(candidate) * projection_dim_ +
                       source_column] -
          maximum);
      denominator += weight;
      output[column] +=
          weight *
          kv_state_[static_cast<std::size_t>(candidate) * projection_dim_ +
                    source_column];
    }
    output[column] /= denominator;
    if (!std::isfinite(output[column])) {
      return Status::InvalidArgument(
          "DeepSeek compressor pooling output is nonfinite");
    }
  }
  if (ratio_ == 4) {
    const auto row_bytes = static_cast<std::size_t>(ratio_) * projection_dim_;
    std::copy_n(kv_state_.begin() + row_bytes, row_bytes, kv_state_.begin());
    std::copy_n(score_state_.begin() + row_bytes, row_bytes,
                score_state_.begin());
  }
  remainder_tokens_ = 0;
  ++completed_slots_;
  return output;
}

}  // namespace pih
