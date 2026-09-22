#include "pih/model/deepseek_sampler.h"

#include <array>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace pih {
namespace {

bool valid_suppression(const DeepSeekSuppressedTokenSet& suppressed,
                       std::size_t vocabulary_size) {
  if (suppressed.token_count > DeepSeekSuppressedTokenSet::kMaximumTokenIds ||
      suppressed.token_count >= vocabulary_size) return false;
  for (std::uint32_t index = 0;
       index < DeepSeekSuppressedTokenSet::kMaximumTokenIds; ++index) {
    if (index >= suppressed.token_count) {
      if (suppressed.token_ids[index] != 0U) return false;
      continue;
    }
    if (suppressed.token_ids[index] >= vocabulary_size) return false;
    for (std::uint32_t prior = 0; prior < index; ++prior)
      if (suppressed.token_ids[prior] == suppressed.token_ids[index])
        return false;
  }
  return true;
}

bool suppressed(const DeepSeekSuppressedTokenSet& suppressed,
                std::uint32_t token_id) {
  for (std::uint32_t index = 0; index < suppressed.token_count; ++index)
    if (suppressed.token_ids[index] == token_id) return true;
  return false;
}

std::pair<std::uint32_t, std::uint32_t> multiply_high_low(
    std::uint32_t left, std::uint32_t right) {
  const auto product = static_cast<std::uint64_t>(left) * right;
  return {static_cast<std::uint32_t>(product >> 32U),
          static_cast<std::uint32_t>(product)};
}

}  // namespace

Result<std::uint32_t> deepseek_greedy_sample(
    std::span<const float> logits,
    const DeepSeekSuppressedTokenSet& suppressed_tokens) {
  if (logits.empty() ||
      logits.size() > std::numeric_limits<std::uint32_t>::max() ||
      !valid_suppression(suppressed_tokens, logits.size())) {
    return Status::InvalidArgument("DeepSeek greedy logits are invalid");
  }
  std::optional<std::uint32_t> best;
  for (std::size_t token = 0; token < logits.size(); ++token) {
    if (!std::isfinite(logits[token])) {
      return Status::Internal("DeepSeek logits contain a non-finite value");
    }
    if (suppressed(suppressed_tokens, static_cast<std::uint32_t>(token)))
      continue;
    if (!best.has_value() || logits[token] > logits[*best]) {
      best = static_cast<std::uint32_t>(token);
    }
  }
  return *best;
}

Result<std::uint32_t> deepseek_philox_word(
    std::uint64_t seed, std::uint64_t sample_ordinal) {
  const auto block = sample_ordinal / 4U;
  std::array<std::uint32_t, 4> counter{
      static_cast<std::uint32_t>(block),
      static_cast<std::uint32_t>(block >> 32U), 0, 0};
  std::uint32_t key0 = static_cast<std::uint32_t>(seed);
  std::uint32_t key1 = static_cast<std::uint32_t>(seed >> 32U);
  for (std::uint32_t round = 0; round < 10; ++round) {
    const auto [high0, low0] =
        multiply_high_low(0xD2511F53U, counter[0]);
    const auto [high1, low1] =
        multiply_high_low(0xCD9E8D57U, counter[2]);
    counter = {high1 ^ counter[1] ^ key0, low1,
               high0 ^ counter[3] ^ key1, low0};
    key0 += 0x9E3779B9U;
    key1 += 0xBB67AE85U;
  }
  return counter[static_cast<std::size_t>(sample_ordinal % 4U)];
}

Result<DeepSeekSamplingResult> deepseek_sample_cpu(
    std::span<const float> logits,
    const DeepSeekSamplingDescriptor& descriptor) {
  if (logits.empty() || !std::isfinite(descriptor.temperature) ||
      descriptor.temperature <= 0.0F || descriptor.temperature > 2.0F ||
      !std::isfinite(descriptor.top_p) || descriptor.top_p <= 0.0F ||
      descriptor.top_p > 1.0F ||
      descriptor.top_logprobs_count > 20U ||
      !valid_suppression(descriptor.suppressed_tokens, logits.size()) ||
      (descriptor.top_k.has_value() &&
       (*descriptor.top_k == 0 || *descriptor.top_k > logits.size()))) {
    return Status::InvalidArgument(
        "DeepSeek stochastic sampling descriptor is invalid");
  }
  std::vector<float> scaled(logits.size());
  for (std::size_t index = 0; index < logits.size(); ++index) {
    if (!std::isfinite(logits[index])) {
      return Status::Internal("DeepSeek logits contain a non-finite value");
    }
    scaled[index] = logits[index] / descriptor.temperature;
  }
  float allowed_maximum = -std::numeric_limits<float>::infinity();
  for (std::size_t index = 0; index < scaled.size(); ++index) {
    if (!suppressed(descriptor.suppressed_tokens,
                    static_cast<std::uint32_t>(index))) {
      allowed_maximum = std::max(allowed_maximum, scaled[index]);
    }
  }
  float full_sum = 0.0F;
  for (std::size_t index = 0; index < scaled.size(); ++index) {
    if (!suppressed(descriptor.suppressed_tokens,
                    static_cast<std::uint32_t>(index))) {
      full_sum += std::exp(scaled[index] - allowed_maximum);
    }
  }
  const auto logsumexp = allowed_maximum + std::log(full_sum);
  std::vector<std::uint32_t> order;
  order.reserve(logits.size());
  for (std::uint32_t id = 0; id < logits.size(); ++id) {
    if (!suppressed(descriptor.suppressed_tokens, id)) order.push_back(id);
  }
  std::stable_sort(order.begin(), order.end(), [&](auto left, auto right) {
    return scaled[left] > scaled[right] ||
           (scaled[left] == scaled[right] && left < right);
  });
  if (descriptor.top_logprobs_count > order.size()) {
    return Status::InvalidArgument(
        "DeepSeek top logprobs exceed allowed vocabulary");
  }
  std::vector<std::uint32_t> top_ids(
      order.begin(), order.begin() + descriptor.top_logprobs_count);
  std::vector<float> top_logprobs;
  top_logprobs.reserve(top_ids.size());
  for (const auto id : top_ids) {
    top_logprobs.push_back(scaled[id] - logsumexp);
  }
  if (descriptor.top_k.has_value() && *descriptor.top_k < order.size()) {
    order.resize(*descriptor.top_k);
  }
  std::vector<float> weights(order.size());
  float filtered_sum = 0.0F;
  for (std::size_t index = 0; index < order.size(); ++index) {
    weights[index] = std::exp(scaled[order[index]] - scaled[order.front()]);
    filtered_sum += weights[index];
  }
  float prefix = 0.0F;
  std::size_t retained = 0;
  do {
    prefix += weights[retained++];
  } while (retained < weights.size() &&
           prefix / filtered_sum < descriptor.top_p);
  float retained_sum = 0.0F;
  for (std::size_t index = 0; index < retained; ++index) {
    retained_sum += weights[index];
  }
  auto word = deepseek_philox_word(descriptor.seed,
                                   descriptor.sample_ordinal);
  if (!word.ok()) return word.status();
  const auto uniform =
      (static_cast<double>(*word) + 0.5) / 4294967296.0;
  double cumulative = 0.0;
  std::size_t selected = retained - 1;
  for (std::size_t index = 0; index < retained; ++index) {
    cumulative += static_cast<double>(weights[index] / retained_sum);
    if (cumulative > uniform) {
      selected = index;
      break;
    }
  }
  const auto token = order[selected];
  return DeepSeekSamplingResult{
      token, scaled[token] - logsumexp, *word,
      std::move(top_ids), std::move(top_logprobs)};
}

}  // namespace pih
