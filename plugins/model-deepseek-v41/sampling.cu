#include "sampling.h"
#include <cuda_runtime.h>

namespace pih::deepseek_v41 {
namespace {
constexpr unsigned kSize = 131072, kVocab = 129280;
template<class T> T* Ptr(EngramDeviceRegion r) { return reinterpret_cast<T*>(r.address); }
__global__ void Initialize(const float* logits, float* scores, unsigned* ids, SamplingCandidate* out,
    unsigned* error, SamplingParameters p) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (!i) { *out = {}; out->token_id = 0xffffffffU; }
  if (i >= kSize) return;
  float score = -CUDART_INF_F;
  if (i < kVocab) {
    const float raw = logits[i];
    if (!isfinite(raw)) atomicOr(error, 1U);
    score = raw / (p.temperature == 0 ? 1.0f : p.temperature);
    if (!isfinite(score)) { atomicOr(error, 2U); score = 0; }
    for (unsigned j = 0; j < p.suppressed_count; ++j) if (p.suppressed[j] == i) score = -CUDART_INF_F;
  }
  scores[i] = score; ids[i] = i < kVocab ? i : 0xffffffffU;
}
__device__ bool Before(float a, unsigned ai, float b, unsigned bi) { return a > b || (a == b && ai < bi); }
__global__ void Sort(float* scores, unsigned* ids, unsigned width, unsigned stride) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x, j = i ^ stride;
  if (i >= kSize || j <= i) return;
  const float a = scores[i], b = scores[j]; const unsigned ai = ids[i], bi = ids[j];
  const bool exchange = (i & width) ? Before(a, ai, b, bi) : Before(b, bi, a, ai);
  if (exchange) { scores[i] = b; scores[j] = a; ids[i] = bi; ids[j] = ai; }
}
__global__ void Weights(const float* scores, float* weights, float* reduction, unsigned allowed, unsigned* error) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= kSize) return;
  const float value = i < allowed ? expf(scores[i] - scores[0]) : 0.0f;
  if (!isfinite(value)) atomicOr(error, 2U);
  weights[i] = value; reduction[i] = value;
}
__global__ void Reduce(float* work, unsigned stride) {
  const unsigned i = (blockIdx.x * blockDim.x + threadIdx.x) * stride * 2;
  if (i < kSize) work[i] += work[i + stride];
}
__global__ void SaveFull(const float* scores, const float* work, float* stats, unsigned* error) {
  const float normalizer = scores[0] + logf(work[0]);
  if (!(work[0] > 0) || !isfinite(normalizer)) atomicOr(error, 2U);
  stats[0] = normalizer;
}
__global__ void Filter(const float* weights, float* work, unsigned count) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < kSize) work[i] = i < count ? weights[i] : 0.0f;
}
__device__ unsigned Philox(unsigned long long seed, unsigned long long ordinal) {
  const auto block = ordinal / 4;
  unsigned c[4]{static_cast<unsigned>(block), static_cast<unsigned>(block >> 32), 0, 0};
  unsigned k0 = static_cast<unsigned>(seed), k1 = static_cast<unsigned>(seed >> 32);
  for (unsigned round = 0; round < 10; ++round) {
    const unsigned long long a = 0xD2511F53ULL * c[0], b = 0xCD9E8D57ULL * c[2];
    const unsigned next0 = static_cast<unsigned>(b >> 32) ^ c[1] ^ k0;
    const unsigned next2 = static_cast<unsigned>(a >> 32) ^ c[3] ^ k1;
    c[0] = next0; c[1] = static_cast<unsigned>(b); c[2] = next2; c[3] = static_cast<unsigned>(a);
    k0 += 0x9E3779B9U; k1 += 0xBB67AE85U;
  }
  return c[ordinal % 4];
}
__global__ void Finalize(const float* scores, const unsigned* ids, const float* weights, const float* work,
    float* stats, SamplingCandidate* out, unsigned* error, unsigned count, SamplingParameters p) {
  if (*error) return;
  const float sum = work[0]; stats[1] = sum;
  if (!(sum > 0) || !isfinite(sum)) { atomicOr(error, 2U); return; }
  unsigned selected = 0, word = 0;
  if (p.temperature != 0) {
    float prefix = 0; unsigned retained = 0;
    do { prefix += weights[retained++]; } while (retained < count && prefix / sum < p.top_p);
    if (!(prefix > 0) || !isfinite(prefix)) { atomicOr(error, 2U); return; }
    word = Philox(p.seed, p.ordinal);
    const double uniform = (static_cast<double>(word) + 0.5) / 4294967296.0;
    double cumulative = 0; selected = retained - 1;
    for (unsigned i = 0; i < retained; ++i) {
      cumulative += static_cast<double>(weights[i] / prefix);
      if (cumulative > uniform) { selected = i; break; }
    }
  }
  SamplingCandidate result{};
  result.token_id = ids[selected]; result.rng_word = word;
  if (p.logprobs) {
    result.selected_logprob = scores[selected] - stats[0]; result.top_count = p.top_count;
    if (!isfinite(result.selected_logprob)) { atomicOr(error, 2U); return; }
    for (unsigned i = 0; i < p.top_count; ++i) {
      result.top_ids[i] = ids[i]; result.top_logprobs[i] = scores[i] - stats[0];
      if (!isfinite(result.top_logprobs[i])) { atomicOr(error, 2U); return; }
    }
  }
  *out = result;
}
Status Last() { const auto e = cudaGetLastError(); return e == cudaSuccess ? Status::Ok() : Status::Internal(cudaGetErrorString(e)); }
}
Status LaunchSampling(const SamplingLaunch& x) {
  const auto validation = ValidateSampling(x); if (!validation.ok()) return validation;
  const auto stream = reinterpret_cast<cudaStream_t>(x.stream);
  auto* scores = Ptr<float>(x.scores); auto* ids = Ptr<unsigned>(x.ids); auto* weights = Ptr<float>(x.weights);
  auto* work = Ptr<float>(x.reduction); auto* error = Ptr<unsigned>(x.error_flag); auto* stats = Ptr<float>(x.stats);
  auto* out = Ptr<SamplingCandidate>(x.candidate); const auto p = x.parameters;
  Initialize<<<kSize / 256, 256, 0, stream>>>(Ptr<const float>(x.logits), scores, ids, out, error, p);
  auto status = Last(); if (!status.ok()) return status;
  for (unsigned width = 2; width <= kSize; width *= 2) for (unsigned stride = width / 2; stride; stride /= 2) {
    Sort<<<kSize / 256, 256, 0, stream>>>(scores, ids, width, stride);
    status = Last(); if (!status.ok()) return status;
  }
  const unsigned allowed = kVocab - p.suppressed_count;
  Weights<<<kSize / 256, 256, 0, stream>>>(scores, weights, work, allowed, error);
  status = Last(); if (!status.ok()) return status;
  for (unsigned stride = 1; stride < kSize; stride *= 2) {
    Reduce<<<(kSize / (stride * 2) + 255) / 256, 256, 0, stream>>>(work, stride);
    status = Last(); if (!status.ok()) return status;
  }
  SaveFull<<<1, 1, 0, stream>>>(scores, work, stats, error);
  status = Last(); if (!status.ok()) return status;
  const unsigned count = p.top_k && p.top_k < allowed ? p.top_k : allowed;
  Filter<<<kSize / 256, 256, 0, stream>>>(weights, work, count);
  status = Last(); if (!status.ok()) return status;
  for (unsigned stride = 1; stride < kSize; stride *= 2) {
    Reduce<<<(kSize / (stride * 2) + 255) / 256, 256, 0, stream>>>(work, stride);
    status = Last(); if (!status.ok()) return status;
  }
  Finalize<<<1, 1, 0, stream>>>(scores, ids, weights, work, stats, out, error, count, p);
  return Last();
}
}  // namespace pih::deepseek_v41
