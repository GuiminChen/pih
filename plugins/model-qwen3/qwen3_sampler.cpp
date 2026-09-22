#include "pih/model/qwen3_sampler.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>

namespace pih { namespace {

std::pair<std::uint32_t, std::uint32_t> multiply_high_low(
    std::uint32_t left, std::uint32_t right) {
  const auto product = static_cast<std::uint64_t>(left) * right;
  return {static_cast<std::uint32_t>(product >> 32U),
          static_cast<std::uint32_t>(product)};
}

Status validate(std::span<const float> logits,
                const Qwen3SamplingDescriptor& descriptor) {
  if (logits.empty() ||
      logits.size() > std::numeric_limits<std::uint32_t>::max() ||
      descriptor.top_logprobs_count > 20U ||
      descriptor.suppressed_token_count >
          Qwen3SamplingDescriptor::kMaximumSuppressedTokenIds ||
      descriptor.suppressed_token_count >= logits.size()) {
    return Status::InvalidArgument("Qwen sampling descriptor is invalid");
  }
  for (std::uint32_t i = 0; i < descriptor.suppressed_token_ids.size(); ++i) {
    if (i >= descriptor.suppressed_token_count) {
      if (descriptor.suppressed_token_ids[i] != 0)
        return Status::InvalidArgument(
            "Qwen suppressed-token tail is not canonical");
      continue;
    }
    if (descriptor.suppressed_token_ids[i] >= logits.size())
      return Status::InvalidArgument(
          "Qwen suppressed token is outside the vocabulary");
    for (std::uint32_t previous = 0; previous < i; ++previous) {
      if (descriptor.suppressed_token_ids[previous] ==
          descriptor.suppressed_token_ids[i])
        return Status::InvalidArgument(
            "Qwen suppressed-token set contains duplicates");
    }
  }
  if (descriptor.top_logprobs_count >
      logits.size() - descriptor.suppressed_token_count) {
    return Status::InvalidArgument(
        "Qwen top-logprobs count exceeds the allowed vocabulary");
  }
  if (descriptor.mode == Qwen3SamplingMode::kGreedy) {
    if (descriptor.temperature != 0.0F || descriptor.top_p != 1.0F ||
        descriptor.top_k.has_value()) {
      return Status::InvalidArgument(
          "Qwen greedy sampling parameters conflict");
    }
  } else if (!std::isfinite(descriptor.temperature) ||
             descriptor.temperature <= 0.0F ||
             descriptor.temperature > 2.0F ||
             !std::isfinite(descriptor.top_p) || descriptor.top_p <= 0.0F ||
             descriptor.top_p > 1.0F ||
             (descriptor.top_k.has_value() &&
              (*descriptor.top_k == 0 ||
               *descriptor.top_k > logits.size()))) {
    return Status::InvalidArgument(
        "Qwen stochastic sampling parameters are invalid");
  }
  return Status::Ok();
}

bool suppressed(const Qwen3SamplingDescriptor& descriptor,
                std::uint32_t token_id) {
  return std::find(
             descriptor.suppressed_token_ids.begin(),
             descriptor.suppressed_token_ids.begin() +
                 descriptor.suppressed_token_count,
             token_id) != descriptor.suppressed_token_ids.begin() +
                             descriptor.suppressed_token_count;
}

} }  // namespace pih::<anonymous>

namespace pih {

Result<std::uint32_t> qwen3_philox_word(
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

Result<Qwen3SamplingResult> qwen3_sample_cpu(
    std::span<const float> logits,
    const Qwen3SamplingDescriptor& descriptor) {
  auto status = validate(logits, descriptor);
  if (!status.ok()) return status;
  const float divisor = descriptor.mode == Qwen3SamplingMode::kGreedy
                            ? 1.0F
                            : descriptor.temperature;
  std::vector<float> scaled(logits.size());
  float allowed_maximum = -std::numeric_limits<float>::infinity();
  for (std::size_t id = 0; id < logits.size(); ++id) {
    if (!std::isfinite(logits[id])) {
      return Status::Internal("Qwen logits contain a non-finite value");
    }
    scaled[id] = logits[id] / divisor;
    if (!suppressed(descriptor, static_cast<std::uint32_t>(id))) {
      allowed_maximum = std::max(allowed_maximum, scaled[id]);
    }
  }
  float full_sum = 0.0F;
  std::vector<std::uint32_t> order;
  order.reserve(logits.size());
  for (std::uint32_t id = 0; id < logits.size(); ++id) {
    if (suppressed(descriptor, id)) continue;
    full_sum += std::exp(scaled[id] - allowed_maximum);
    order.push_back(id);
  }
  const auto logsumexp = allowed_maximum + std::log(full_sum);
  std::stable_sort(order.begin(), order.end(), [&](auto left, auto right) {
    return scaled[left] > scaled[right] ||
           (scaled[left] == scaled[right] && left < right);
  });
  std::vector<std::uint32_t> top_ids(
      order.begin(), order.begin() + descriptor.top_logprobs_count);
  std::vector<float> top_logprobs;
  top_logprobs.reserve(top_ids.size());
  for (const auto id : top_ids) {
    top_logprobs.push_back(scaled[id] - logsumexp);
  }
  if (descriptor.mode == Qwen3SamplingMode::kGreedy) {
    const auto token = order.front();
    return Qwen3SamplingResult{token, scaled[token] - logsumexp, 0,
                               std::move(top_ids),
                               std::move(top_logprobs)};
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
  float retained_sum = std::accumulate(
      weights.begin(), weights.begin() + retained, 0.0F);
  auto word = qwen3_philox_word(descriptor.seed, descriptor.sample_ordinal);
  if (!word.ok()) return word.status();
  const auto uniform =
      (static_cast<double>(*word) + 0.5) / 4294967296.0;
  double cumulative = 0.0;
  std::size_t selected = retained - 1U;
  for (std::size_t index = 0; index < retained; ++index) {
    cumulative += static_cast<double>(weights[index] / retained_sum);
    if (cumulative > uniform) {
      selected = index;
      break;
    }
  }
  const auto token = order[selected];
  return Qwen3SamplingResult{token, scaled[token] - logsumexp, *word,
                             std::move(top_ids),
                             std::move(top_logprobs)};
}

}  // namespace pih
