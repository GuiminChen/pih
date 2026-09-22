#include "sampling.h"
#include <array>
#include <cmath>
#include <limits>
#include <bit>

namespace pih::deepseek_v41 {
Status ValidateSamplingParameters(const SamplingParameters& p) {
  if (!std::isfinite(p.temperature) || p.temperature < 0 || p.temperature > 2 ||
      !std::isfinite(p.top_p) || p.top_p <= 0 || p.top_p > 1 || p.top_k > 129280 ||
      (p.temperature == 0 && (p.top_p != 1 || p.top_k)) || p.top_count > 20 ||
      (!p.logprobs && p.top_count) || p.suppressed_count > 17)
    return Status::InvalidArgument("V4.1 sampling parameters are not canonical");
  for (unsigned i = 0; i < 17; ++i) {
    if (i >= p.suppressed_count) {
      if (p.suppressed[i]) return Status::InvalidArgument("Sampling suppression padding must be zero");
    } else {
      if (p.suppressed[i] >= 129280) return Status::InvalidArgument("Suppressed token is out of vocabulary");
      for (unsigned j = 0; j < i; ++j) if (p.suppressed[j] == p.suppressed[i])
        return Status::InvalidArgument("Duplicate suppressed token");
    }
  }
  return Status::Ok();
}
Status ValidateSampling(const SamplingLaunch& x) {
  const auto parameters = ValidateSamplingParameters(x.parameters); if (!parameters.ok()) return parameters;
  if (!x.stream) return Status::InvalidArgument("Sampling requires an explicit stream");
  const std::array regions{x.logits, x.scores, x.ids, x.weights, x.reduction, x.stats, x.candidate, x.error_flag};
  const std::array<std::uint64_t, 8> bytes{129280ULL * 4, 131072ULL * 4, 131072ULL * 4,
      131072ULL * 4, 131072ULL * 4, 8, sizeof(SamplingCandidate), 4};
  for (unsigned i = 0; i < regions.size(); ++i) {
    const auto r = regions[i];
    if (!r.address || r.address % 4 || r.bytes != bytes[i] || r.bytes > std::numeric_limits<std::uintptr_t>::max() - r.address)
      return Status::InvalidArgument("Sampling storage extent or alignment invalid");
    for (unsigned j = 0; j < i; ++j)
      if (r.address < regions[j].address + regions[j].bytes && regions[j].address < r.address + r.bytes)
        return Status::InvalidArgument("Sampling buffers must be disjoint");
  }
  return Status::Ok();
}
std::uint32_t SamplingPhiloxWord(std::uint64_t seed, std::uint64_t ordinal) noexcept {
  const auto block = ordinal / 4;
  std::array<std::uint32_t, 4> counter{static_cast<std::uint32_t>(block), static_cast<std::uint32_t>(block >> 32), 0, 0};
  auto k0 = static_cast<std::uint32_t>(seed), k1 = static_cast<std::uint32_t>(seed >> 32);
  for (unsigned round = 0; round < 10; ++round) {
    const std::uint64_t a = 0xD2511F53ULL * counter[0], b = 0xCD9E8D57ULL * counter[2];
    counter = {static_cast<std::uint32_t>(b >> 32) ^ counter[1] ^ k0, static_cast<std::uint32_t>(b),
        static_cast<std::uint32_t>(a >> 32) ^ counter[3] ^ k1, static_cast<std::uint32_t>(a)};
    k0 += 0x9E3779B9U; k1 += 0xBB67AE85U;
  }
  return counter[ordinal % 4];
}
Status ValidateSamplingCandidate(const SamplingCandidate& x, const SamplingParameters& p) {
  const auto parameters = ValidateSamplingParameters(p); if (!parameters.ok()) return parameters;
  const auto allowed = [&](std::uint32_t id) {
    if (id >= 129280) return false;
    for (unsigned i = 0; i < p.suppressed_count; ++i) if (id == p.suppressed[i]) return false;
    return true;
  };
  const auto word = p.temperature == 0 ? 0U : SamplingPhiloxWord(p.seed, p.ordinal);
  if (!allowed(x.token_id) || x.reserved || x.rng_word != word || x.top_count != p.top_count ||
      !std::isfinite(x.selected_logprob) || x.selected_logprob > 0 ||
      (!p.logprobs && std::bit_cast<std::uint32_t>(x.selected_logprob) != 0))
    return Status::FailedPrecondition("Sampling candidate token, RNG or logprob metadata invalid");
  for (unsigned i = 0; i < 20; ++i) {
    if (i >= x.top_count) {
      if (x.top_ids[i] || std::bit_cast<std::uint32_t>(x.top_logprobs[i]) != 0)
        return Status::FailedPrecondition("Sampling candidate unused alternatives are not zero");
      continue;
    }
    if (!allowed(x.top_ids[i]) || !std::isfinite(x.top_logprobs[i]) || x.top_logprobs[i] > 0 ||
        (i && x.top_logprobs[i] > x.top_logprobs[i - 1]) ||
        (x.top_ids[i] == x.token_id && x.top_logprobs[i] != x.selected_logprob))
      return Status::FailedPrecondition("Sampling alternative token or logprob invalid");
    for (unsigned j = 0; j < i; ++j) if (x.top_ids[i] == x.top_ids[j])
      return Status::FailedPrecondition("Sampling alternatives contain duplicate tokens");
  }
  // Rounded logprobs may tie even when scaled logits differ, so their ties do
  // not prove token-ID ordering. Original-score ordering needs GPU fixtures.
  if (p.temperature == 0 && x.top_count && x.token_id != x.top_ids[0])
    return Status::FailedPrecondition("Greedy candidate differs from highest alternative");
  return Status::Ok();
}
}  // namespace pih::deepseek_v41
