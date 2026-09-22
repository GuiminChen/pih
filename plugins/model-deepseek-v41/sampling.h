#pragma once
#include "engram_launch.h"

namespace pih::deepseek_v41 {
struct SamplingParameters final {
  float temperature = 0, top_p = 1;
  std::uint32_t top_k = 0;  // Zero means the complete allowed vocabulary.
  std::uint64_t seed = 0, ordinal = 0;
  bool logprobs = false;
  std::uint32_t top_count = 0, suppressed_count = 0, suppressed[17]{};
};
struct SamplingCandidate final {
  std::uint32_t token_id, rng_word, top_count, reserved;
  float selected_logprob;
  std::uint32_t top_ids[20];
  float top_logprobs[20];
};
static_assert(sizeof(SamplingCandidate) == 180);
struct SamplingIdentity final {
  std::uint64_t epoch = 0, plan_seq = 0, sequence_generation = 0, sampling_config_id = 0;
};
struct SamplingObservation final {
  SamplingIdentity identity;
  std::uint64_t ordinal = 0;
  std::uint32_t processed_length = 0;
  SamplingCandidate candidate{};
};
struct SamplingLaunch final {
  // FP32 logits [129280]; sorted scores/IDs, weights and reduction scratch
  // each [131072] with four-byte elements; stats [2] FP32; fixed candidate.
  EngramDeviceRegion logits, scores, ids, weights, reduction, stats, candidate, error_flag;
  std::uintptr_t stream = 0;
  SamplingParameters parameters;
};
Status ValidateSampling(const SamplingLaunch& launch);
Status ValidateSamplingParameters(const SamplingParameters& parameters);
std::uint32_t SamplingPhiloxWord(std::uint64_t seed, std::uint64_t ordinal) noexcept;
// Structural validation of a completed device observation, not proof of logits
// authenticity or controller acceptance.
Status ValidateSamplingCandidate(const SamplingCandidate& candidate, const SamplingParameters& parameters);
Status LaunchSampling(const SamplingLaunch& launch);
}  // namespace pih::deepseek_v41
