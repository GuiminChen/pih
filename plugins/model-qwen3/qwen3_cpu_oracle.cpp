#include "pih/model/qwen3_cpu_oracle.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "pih/core/checked_math.h"

namespace pih {
namespace {

Result<std::size_t> extent(std::uint64_t first, std::uint64_t second) {
  auto product = checked_mul_u64(first, second);
  if (!product.ok()) return product.status();
  if (product.value() >
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return Status::ResourceExhausted("oracle extent exceeds address space");
  }
  return static_cast<std::size_t>(product.value());
}

Status publish(std::span<const float> values, std::span<BFloat16> output) {
  if (values.size() != output.size()) {
    return Status::InvalidArgument("oracle output extent mismatch");
  }
  std::vector<BFloat16> converted;
  converted.reserve(values.size());
  for (const float value : values) {
    if (!std::isfinite(value)) {
      return Status::FailedPrecondition("oracle produced a nonfinite value");
    }
    converted.push_back(BFloat16::FromFloat(value));
  }
  std::copy(converted.begin(), converted.end(), output.begin());
  return Status::Ok();
}

}  // namespace

Status qwen_embedding_oracle(std::span<const BFloat16> table,
                             std::uint64_t vocabulary_size,
                             std::uint64_t hidden_size,
                             std::span<const std::int64_t> token_ids,
                             std::span<BFloat16> output) {
  auto table_elements = extent(vocabulary_size, hidden_size);
  auto output_elements = extent(token_ids.size(), hidden_size);
  if (!table_elements.ok()) return table_elements.status();
  if (!output_elements.ok()) return output_elements.status();
  if (table.size() != table_elements.value() ||
      output.size() != output_elements.value()) {
    return Status::InvalidArgument("embedding extent mismatch");
  }
  for (const auto token : token_ids) {
    if (token < 0 || static_cast<std::uint64_t>(token) >= vocabulary_size) {
      return Status::InvalidArgument("token id is outside the vocabulary");
    }
  }
  std::vector<BFloat16> result(output.size());
  for (std::size_t row = 0; row < token_ids.size(); ++row) {
    const auto source = static_cast<std::size_t>(token_ids[row]) * hidden_size;
    const auto destination = row * hidden_size;
    std::copy_n(table.begin() + source, static_cast<std::size_t>(hidden_size),
                result.begin() + destination);
  }
  std::copy(result.begin(), result.end(), output.begin());
  return Status::Ok();
}

Status qwen_rms_norm_oracle(std::span<const BFloat16> input,
                            std::span<const BFloat16> weight,
                            std::uint64_t rows, std::uint64_t hidden_size,
                            float epsilon, std::span<BFloat16> output) {
  auto elements = extent(rows, hidden_size);
  if (!elements.ok()) return elements.status();
  if (hidden_size == 0 || !std::isfinite(epsilon) || epsilon <= 0.0F) {
    return Status::InvalidArgument("RMSNorm requires hidden size and positive epsilon");
  }
  if (input.size() != elements.value() || output.size() != elements.value() ||
      weight.size() != hidden_size) {
    return Status::InvalidArgument("RMSNorm extent mismatch");
  }
  std::vector<float> result(input.size());
  for (std::size_t row = 0; row < static_cast<std::size_t>(rows); ++row) {
    float sum_of_squares = 0.0F;
    const std::size_t offset = row * static_cast<std::size_t>(hidden_size);
    for (std::size_t column = 0; column < hidden_size; ++column) {
      const float value = input[offset + column].to_float();
      sum_of_squares += value * value;
    }
    const float scale = 1.0F / std::sqrt(sum_of_squares / hidden_size + epsilon);
    for (std::size_t column = 0; column < hidden_size; ++column) {
      result[offset + column] = input[offset + column].to_float() * scale *
                                weight[column].to_float();
    }
  }
  return publish(result, output);
}

Status qwen_residual_add_oracle(std::span<const BFloat16> lhs,
                                std::span<const BFloat16> rhs,
                                std::span<BFloat16> output) {
  if (lhs.size() != rhs.size() || lhs.size() != output.size()) {
    return Status::InvalidArgument("residual add extent mismatch");
  }
  std::vector<float> result(lhs.size());
  for (std::size_t index = 0; index < lhs.size(); ++index) {
    result[index] = lhs[index].to_float() + rhs[index].to_float();
  }
  return publish(result, output);
}

Status qwen_silu_mul_oracle(std::span<const BFloat16> gate,
                            std::span<const BFloat16> up,
                            std::span<BFloat16> output) {
  if (gate.size() != up.size() || gate.size() != output.size()) {
    return Status::InvalidArgument("SwiGLU extent mismatch");
  }
  std::vector<float> result(gate.size());
  for (std::size_t index = 0; index < gate.size(); ++index) {
    const float value = gate[index].to_float();
    result[index] = (value / (1.0F + std::exp(-value))) * up[index].to_float();
  }
  return publish(result, output);
}

Status qwen_rope_angles_oracle(std::span<const std::int64_t> positions,
                               std::uint64_t head_dim, double theta,
                               std::span<float> cosine,
                               std::span<float> sine) {
  if (positions.empty() || head_dim != 128 || theta != 1'000'000.0) {
    return Status::InvalidArgument("Qwen RoPE angle geometry is invalid");
  }
  auto values = extent(positions.size(), head_dim / 2);
  if (!values.ok() || cosine.size() != *values || sine.size() != *values) {
    return Status::InvalidArgument("Qwen RoPE angle output extent mismatch");
  }
  for (const auto position : positions) {
    if (position < 0 || position >= 40960) {
      return Status::InvalidArgument("Qwen RoPE position is outside context");
    }
  }

  std::vector<float> next_cosine(*values);
  std::vector<float> next_sine(*values);
  for (std::size_t token = 0; token < positions.size(); ++token) {
    for (std::uint64_t pair = 0; pair < head_dim / 2; ++pair) {
      const double exponent = -2.0 * static_cast<double>(pair) /
                              static_cast<double>(head_dim);
      const double angle = static_cast<double>(positions[token]) *
                           std::pow(theta, exponent);
      const auto index = token * (head_dim / 2) + pair;
      next_cosine[index] = static_cast<float>(std::cos(angle));
      next_sine[index] = static_cast<float>(std::sin(angle));
    }
  }
  std::copy(next_cosine.begin(), next_cosine.end(), cosine.begin());
  std::copy(next_sine.begin(), next_sine.end(), sine.begin());
  return Status::Ok();
}

Status qwen_rope_oracle(std::span<const BFloat16> input,
                        std::span<const float> cosine,
                        std::span<const float> sine,
                        std::uint64_t tokens, std::uint64_t heads,
                        std::uint64_t head_dim,
                        std::span<BFloat16> output) {
  if (tokens == 0 || heads == 0 || head_dim == 0 || (head_dim & 1U) != 0) {
    return Status::InvalidArgument("RoPE head dimension must be positive and even");
  }
  auto vectors = checked_mul_u64(tokens, heads);
  if (!vectors.ok()) return vectors.status();
  auto elements = extent(*vectors, head_dim);
  auto pairs = extent(tokens, head_dim / 2);
  if (!elements.ok()) return elements.status();
  if (!pairs.ok()) return pairs.status();
  if (input.size() != elements.value() || output.size() != elements.value() ||
      cosine.size() != pairs.value() || sine.size() != pairs.value()) {
    return Status::InvalidArgument("RoPE extent mismatch");
  }
  std::vector<float> result(input.size());
  const std::size_t half = static_cast<std::size_t>(head_dim / 2);
  for (std::size_t vector = 0; vector < *vectors; ++vector) {
    const std::size_t vector_offset = vector * static_cast<std::size_t>(head_dim);
    const std::size_t angle_offset = (vector / heads) * half;
    for (std::size_t pair = 0; pair < half; ++pair) {
      const float first = input[vector_offset + pair].to_float();
      const float second = input[vector_offset + half + pair].to_float();
      const float cos_value = cosine[angle_offset + pair];
      const float sin_value = sine[angle_offset + pair];
      result[vector_offset + pair] = first * cos_value - second * sin_value;
      result[vector_offset + half + pair] = second * cos_value + first * sin_value;
    }
  }
  return publish(result, output);
}

Result<std::int64_t> qwen_greedy_argmax_oracle(
    std::span<const float> logits) {
  if (logits.size() != 151936) {
    return Status::InvalidArgument(
        "Qwen greedy argmax requires the exact vocabulary row");
  }
  float best_value = -std::numeric_limits<float>::infinity();
  std::int64_t best_token = 0;
  for (std::size_t token = 0; token < logits.size(); ++token) {
    if (!std::isfinite(logits[token])) {
      return Status::FailedPrecondition(
          "Qwen greedy argmax rejects nonfinite logits");
    }
    if (logits[token] > best_value) {
      best_value = logits[token];
      best_token = static_cast<std::int64_t>(token);
    }
  }
  return best_token;
}

}  // namespace pih
