#include "pih/model/deepseek_dspark_prefill_reference.h"

#include <cmath>
#include <numeric>

namespace pih {
namespace {

using Receipt = DeepSeekDsparkPrefillReferenceReceipt;

float main_value(std::uint32_t position, std::uint32_t column) {
  return BFloat16::FromFloat(
      0.07F * static_cast<float>(position + 1) +
      0.013F * static_cast<float>(column + 1)).to_float();
}

BFloat16 project_kv(std::uint32_t stage, std::uint32_t position,
                    std::uint32_t output) {
  float value = 0.0F;
  for (std::uint32_t input = 0; input < Receipt::kMainHiddenSize; ++input) {
    const auto sign = (stage + output + input) % 2 == 0 ? 1.0F : -1.0F;
    const auto weight =
        0.04F * static_cast<float>((stage + 1) * (output + 1)) +
        sign * 0.015F * static_cast<float>(input + 1);
    value += main_value(position, input) * weight;
  }
  return BFloat16::FromFloat(value);
}

std::array<BFloat16, Receipt::kKvSize> stage_row(
    std::uint32_t stage, std::uint32_t position) {
  std::array<float, Receipt::kKvSize> projected{};
  float square_sum = 0.0F;
  for (std::uint32_t output = 0; output < Receipt::kKvSize; ++output) {
    projected[output] = project_kv(stage, position, output).to_float();
    square_sum += projected[output] * projected[output];
  }
  const auto inverse_rms = 1.0F / std::sqrt(
      square_sum / static_cast<float>(Receipt::kKvSize) + 1.0e-6F);
  std::array<BFloat16, Receipt::kKvSize> normalized{};
  for (std::uint32_t output = 0; output < Receipt::kKvSize; ++output) {
    const auto scale = 0.9F + 0.05F * static_cast<float>(stage + output);
    normalized[output] =
        BFloat16::FromFloat(projected[output] * inverse_rms * scale);
  }
  const auto angle = 0.125F * static_cast<float>(position);
  const auto cosine = std::cos(angle);
  const auto sine = std::sin(angle);
  const auto left = normalized[2].to_float();
  const auto right = normalized[3].to_float();
  normalized[2] = BFloat16::FromFloat(left * cosine - right * sine);
  normalized[3] = BFloat16::FromFloat(left * sine + right * cosine);
  return normalized;
}

}  // namespace

Result<DeepSeekDsparkPrefillReferenceReceipt>
build_deepseek_dspark_prefill_reference(
    std::span<const std::uint32_t> chunk_sizes) {
  std::array<std::uint32_t, 1> whole{Receipt::kPromptTokens};
  if (chunk_sizes.empty()) chunk_sizes = whole;
  std::uint32_t total = 0;
  for (const auto size : chunk_sizes) {
    if (size == 0 || size > Receipt::kPromptTokens - total) {
      return Status::InvalidArgument(
          "DeepSeek DSpark prefill reference chunk geometry is invalid");
    }
    total += size;
  }
  if (total != Receipt::kPromptTokens) {
    return Status::InvalidArgument(
        "DeepSeek DSpark prefill chunks do not cover the prompt");
  }

  Receipt receipt;
  for (auto& state : receipt.physical_recent_state) {
    state.resize(Receipt::kWindowSize * Receipt::kKvSize);
  }
  std::uint32_t first_position = 0;
  for (const auto chunk : chunk_sizes) {
    for (std::uint32_t offset = 0; offset < chunk; ++offset) {
      const auto position = first_position + offset;
      const auto slot = position % Receipt::kWindowSize;
      for (std::uint32_t stage = 0;
           stage < kDeepSeekDsparkStageCount; ++stage) {
        const auto row = stage_row(stage, position);
        for (std::uint32_t column = 0; column < Receipt::kKvSize; ++column) {
          receipt.physical_recent_state[stage]
              [static_cast<std::size_t>(slot) * Receipt::kKvSize + column] =
                  row[column];
        }
        ++receipt.write_counts[stage];
      }
    }
    first_position += chunk;
  }

  const auto logical_first = Receipt::kPromptTokens - Receipt::kWindowSize;
  for (std::uint32_t row = 0; row < Receipt::kWindowSize; ++row) {
    const auto position = logical_first + row;
    receipt.logical_positions[row] = position;
    const auto slot = position % Receipt::kWindowSize;
    for (std::uint32_t stage = 0;
         stage < kDeepSeekDsparkStageCount; ++stage) {
      auto& logical = receipt.logical_recent_state[stage];
      logical.insert(
          logical.end(),
          receipt.physical_recent_state[stage].begin() +
              static_cast<std::ptrdiff_t>(slot * Receipt::kKvSize),
          receipt.physical_recent_state[stage].begin() +
              static_cast<std::ptrdiff_t>((slot + 1) * Receipt::kKvSize));
    }
  }
  return receipt;
}

}  // namespace pih
