#include "pih/model/deepseek_endpoint_oracle.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace pih { namespace {
constexpr std::uint32_t kHc = 4;
bool finite(std::span<const float> values) {
  for (const auto value : values) if (!std::isfinite(value)) return false;
  return true;
}
}  // namespace pih::<anonymous>

Status deepseek_embedding_expand_oracle(
    std::span<const std::uint32_t> ids, std::span<const BFloat16> weight,
    std::uint32_t vocab, std::uint32_t hidden,
    std::span<BFloat16> output) {
  const auto weight_elements = static_cast<std::size_t>(vocab) * hidden;
  const auto output_elements = ids.size() * kHc * hidden;
  if (ids.empty() || vocab == 0 || hidden == 0 ||
      weight.size() != weight_elements || output.size() != output_elements) {
    return Status::InvalidArgument("DeepSeek embedding oracle shape is invalid");
  }
  for (const auto id : ids) if (id >= vocab)
    return Status::InvalidArgument("DeepSeek token id is out of range");
  std::vector<BFloat16> result(output_elements);
  for (std::size_t token = 0; token < ids.size(); ++token)
    for (std::uint32_t stream = 0; stream < kHc; ++stream)
      std::copy_n(weight.begin() + static_cast<std::size_t>(ids[token]) * hidden,
                  hidden, result.begin() + (token * kHc + stream) * hidden);
  std::copy(result.begin(), result.end(), output.begin());
  return Status::Ok();
}

Status deepseek_hc_head_oracle(
    std::span<const BFloat16> input, std::span<const float> fn,
    std::span<const float> scale, std::span<const float> base,
    std::uint32_t tokens, std::uint32_t hidden, float rms_epsilon,
    float hc_epsilon, std::span<BFloat16> output) {
  const auto width = static_cast<std::size_t>(kHc) * hidden;
  if (tokens == 0 || hidden == 0 ||
      input.size() != static_cast<std::size_t>(tokens) * width ||
      fn.size() != kHc * width || scale.size() != 1 || base.size() != kHc ||
      output.size() != static_cast<std::size_t>(tokens) * hidden ||
      !finite(fn) || !finite(scale) || !finite(base) ||
      !std::isfinite(rms_epsilon) || !std::isfinite(hc_epsilon) ||
      rms_epsilon <= 0.0F || hc_epsilon < 0.0F) {
    return Status::InvalidArgument("DeepSeek HC head oracle is invalid");
  }
  std::vector<BFloat16> result(output.size());
  for (std::uint32_t token = 0; token < tokens; ++token) {
    const auto offset = static_cast<std::size_t>(token) * width;
    float squares = 0.0F;
    for (std::size_t column = 0; column < width; ++column) {
      const auto value = input[offset + column].to_float();
      if (!std::isfinite(value))
        return Status::InvalidArgument("DeepSeek HC head input is non-finite");
      squares += value * value;
    }
    const auto inverse = 1.0F / std::sqrt(squares / width + rms_epsilon);
    float gate[kHc];
    for (std::uint32_t stream = 0; stream < kHc; ++stream) {
      float dot = 0.0F;
      for (std::size_t column = 0; column < width; ++column)
        dot += input[offset + column].to_float() *
               fn[static_cast<std::size_t>(stream) * width + column];
      const auto logit = dot * inverse * scale[0] + base[stream];
      const auto e = std::exp(logit >= 0.0F ? -logit : logit);
      gate[stream] = (logit >= 0.0F ? 1.0F / (1.0F + e)
                                    : e / (1.0F + e)) + hc_epsilon;
    }
    for (std::uint32_t column = 0; column < hidden; ++column) {
      float sum = 0.0F;
      for (std::uint32_t stream = 0; stream < kHc; ++stream)
        sum += gate[stream] * input[offset +
            static_cast<std::size_t>(stream) * hidden + column].to_float();
      result[static_cast<std::size_t>(token) * hidden + column] =
          BFloat16::FromFloat(sum);
    }
  }
  std::copy(result.begin(), result.end(), output.begin());
  return Status::Ok();
}

Status deepseek_lm_head_oracle(
    std::span<const BFloat16> input, std::span<const BFloat16> weight,
    std::uint32_t rows, std::uint32_t vocab, std::uint32_t hidden,
    std::span<float> logits) {
  if (rows == 0 || vocab == 0 || hidden == 0 ||
      input.size() != static_cast<std::size_t>(rows) * hidden ||
      weight.size() != static_cast<std::size_t>(vocab) * hidden ||
      logits.size() != static_cast<std::size_t>(rows) * vocab) {
    return Status::InvalidArgument("DeepSeek LM head oracle shape is invalid");
  }
  std::vector<float> result(logits.size());
  for (std::uint32_t row = 0; row < rows; ++row)
    for (std::uint32_t word = 0; word < vocab; ++word) {
      float sum = 0.0F;
      for (std::uint32_t column = 0; column < hidden; ++column) {
        const auto x = input[static_cast<std::size_t>(row) * hidden + column].to_float();
        const auto w = weight[static_cast<std::size_t>(word) * hidden + column].to_float();
        if (!std::isfinite(x) || !std::isfinite(w))
          return Status::InvalidArgument("DeepSeek LM head input is non-finite");
        sum += x * w;
      }
      result[static_cast<std::size_t>(row) * vocab + word] = sum;
    }
  std::copy(result.begin(), result.end(), logits.begin());
  return Status::Ok();
}

Status deepseek_dspark_markov_oracle(
    std::span<const std::uint32_t> ids,
    std::span<const BFloat16> embedding,
    std::span<const BFloat16> head, std::span<const float> raw,
    std::uint32_t vocab, std::uint32_t rank,
    std::span<BFloat16> output_embedding, std::span<float> biased) {
  const auto rows = ids.size();
  if (rows == 0 || rows > 5 || vocab == 0 || rank == 0 ||
      embedding.size() != static_cast<std::size_t>(vocab) * rank ||
      head.size() != static_cast<std::size_t>(vocab) * rank ||
      raw.size() != rows * vocab || output_embedding.size() != rows * rank ||
      biased.size() != raw.size()) {
    return Status::InvalidArgument("DeepSeek DSpark Markov oracle is invalid");
  }
  for (const auto id : ids) if (id >= vocab)
    return Status::InvalidArgument("DeepSeek DSpark Markov token is invalid");
  std::vector<BFloat16> embeds(output_embedding.size());
  std::vector<float> logits(biased.size());
  for (std::size_t row = 0; row < rows; ++row) {
    std::copy_n(embedding.begin() + static_cast<std::size_t>(ids[row]) * rank,
                rank, embeds.begin() + row * rank);
    for (std::uint32_t word = 0; word < vocab; ++word) {
      float sum = raw[row * vocab + word];
      if (!std::isfinite(sum))
        return Status::InvalidArgument("DeepSeek DSpark raw logit is non-finite");
      for (std::uint32_t column = 0; column < rank; ++column) {
        const auto x = embeds[row * rank + column].to_float();
        const auto w = head[static_cast<std::size_t>(word) * rank + column]
                           .to_float();
        if (!std::isfinite(x) || !std::isfinite(w))
          return Status::InvalidArgument(
              "DeepSeek DSpark Markov operand is non-finite");
        sum += x * w;
      }
      if (!std::isfinite(sum))
        return Status::InvalidArgument("DeepSeek DSpark biased logit is non-finite");
      logits[row * vocab + word] = sum;
    }
  }
  std::copy(embeds.begin(), embeds.end(), output_embedding.begin());
  std::copy(logits.begin(), logits.end(), biased.begin());
  return Status::Ok();
}

Status deepseek_dspark_confidence_oracle(
    std::span<const BFloat16> hidden,
    std::span<const BFloat16> markov, std::span<const float> weight,
    std::uint32_t rows, std::uint32_t hidden_size, std::uint32_t rank,
    std::span<float> confidence) {
  if (rows == 0 || rows > 5 || hidden_size == 0 || rank == 0 ||
      hidden.size() != static_cast<std::size_t>(rows) * hidden_size ||
      markov.size() != static_cast<std::size_t>(rows) * rank ||
      weight.size() != hidden_size + rank || confidence.size() != rows ||
      !finite(weight)) {
    return Status::InvalidArgument(
        "DeepSeek DSpark confidence oracle is invalid");
  }
  std::vector<float> result(rows);
  for (std::uint32_t row = 0; row < rows; ++row) {
    float sum = 0.0F;
    for (std::uint32_t column = 0; column < hidden_size; ++column) {
      const auto value = hidden[static_cast<std::size_t>(row) * hidden_size +
                                column].to_float();
      if (!std::isfinite(value))
        return Status::InvalidArgument(
            "DeepSeek DSpark confidence hidden is non-finite");
      sum += value * weight[column];
    }
    for (std::uint32_t column = 0; column < rank; ++column) {
      const auto value = markov[static_cast<std::size_t>(row) * rank + column]
                             .to_float();
      if (!std::isfinite(value))
        return Status::InvalidArgument(
            "DeepSeek DSpark confidence Markov value is non-finite");
      sum += value * weight[hidden_size + column];
    }
    if (!std::isfinite(sum))
      return Status::InvalidArgument(
          "DeepSeek DSpark confidence result is non-finite");
    result[row] = sum;
  }
  std::copy(result.begin(), result.end(), confidence.begin());
  return Status::Ok();
}

Result<std::uint32_t> deepseek_argmax_oracle(
    std::span<const float> logits) {
  if (logits.size() != 129280)
    return Status::InvalidArgument("DeepSeek DSpark argmax shape is invalid");
  std::uint32_t best = 0;
  if (!std::isfinite(logits[0]))
    return Status::InvalidArgument("DeepSeek DSpark argmax logit is non-finite");
  for (std::uint32_t id = 1; id < logits.size(); ++id) {
    if (!std::isfinite(logits[id]))
      return Status::InvalidArgument(
          "DeepSeek DSpark argmax logit is non-finite");
    if (logits[id] > logits[best]) best = id;
  }
  return best;
}

Status deepseek_dspark_draft_init_oracle(
    std::span<const std::uint32_t> input,
    std::span<const BFloat16> embedding, std::uint32_t noise,
    std::uint32_t block, std::uint32_t vocab, std::uint32_t hidden,
    std::span<std::uint32_t> draft, std::span<BFloat16> output) {
  const auto sequences = input.size();
  const auto draft_count = sequences * block;
  const auto output_count = draft_count * kHc * hidden;
  if (sequences == 0 || block != 5 || vocab == 0 || hidden == 0 ||
      noise >= vocab ||
      embedding.size() != static_cast<std::size_t>(vocab) * hidden ||
      draft.size() != draft_count || output.size() != output_count) {
    return Status::InvalidArgument(
        "DeepSeek DSpark draft init oracle is invalid");
  }
  for (const auto id : input) if (id >= vocab)
    return Status::InvalidArgument(
        "DeepSeek DSpark draft input token is invalid");
  std::vector<std::uint32_t> ids(draft_count);
  std::vector<BFloat16> hc(output_count);
  for (std::size_t sequence = 0; sequence < sequences; ++sequence) {
    for (std::uint32_t position = 0; position < block; ++position) {
      const auto id = position == 0 ? input[sequence] : noise;
      const auto row = sequence * block + position;
      ids[row] = id;
      for (std::uint32_t stream = 0; stream < kHc; ++stream) {
        std::copy_n(embedding.begin() + static_cast<std::size_t>(id) * hidden,
                    hidden, hc.begin() + (row * kHc + stream) * hidden);
      }
    }
  }
  std::copy(ids.begin(), ids.end(), draft.begin());
  std::copy(hc.begin(), hc.end(), output.begin());
  return Status::Ok();
}

}  // namespace pih
