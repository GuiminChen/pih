#include "indexer_select.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace pih::deepseek_v41 {
namespace {
template<class T> T* Ptr(EngramDeviceRegion x) { return reinterpret_cast<T*>(x.address); }
__device__ bool Better(float a, unsigned ai, float b, unsigned bi) {
  return a > b || (a == b && ai < bi);
}
__device__ void Insert(float* heap_scores, unsigned* heap_indices, unsigned& count,
    unsigned picks, float score, unsigned position) {
  if (count < picks) {
    unsigned child = count++;
    while (child) {
      const unsigned parent = (child - 1) / 2;
      if (!Better(heap_scores[parent], heap_indices[parent], score, position)) break;
      heap_scores[child] = heap_scores[parent]; heap_indices[child] = heap_indices[parent]; child = parent;
    }
    heap_scores[child] = score; heap_indices[child] = position;
  } else if (Better(score, position, heap_scores[0], heap_indices[0])) {
    unsigned parent = 0;
    while (parent * 2 + 1 < picks) {
      unsigned child = parent * 2 + 1;
      if (child + 1 < picks && Better(heap_scores[child], heap_indices[child], heap_scores[child + 1], heap_indices[child + 1])) ++child;
      if (!Better(score, position, heap_scores[child], heap_indices[child])) break;
      heap_scores[parent] = heap_scores[child]; heap_indices[parent] = heap_indices[child]; parent = child;
    }
    heap_scores[parent] = score; heap_indices[parent] = position;
  }
}
// A bounded worst-first heap retains the best 512 scores without per-position
// scratch. This initial serial-per-query kernel needs throughput qualification.
__global__ void Select(const __nv_bfloat16* scores, const unsigned char* candidates, int* output, unsigned* error,
    unsigned start, unsigned ratio, unsigned positions, unsigned offset) {
  __shared__ float heap_scores[512];
  __shared__ unsigned heap_indices[512];
  const unsigned token = blockIdx.x, picks = positions < 512 ? positions : 512;
  const unsigned visible = (start + token + 1) / ratio;
  unsigned count = 0;
  for (unsigned position = 0; position < positions; ++position) {
    float score = __bfloat162float(scores[static_cast<unsigned long long>(token) * positions + position]);
    if (!isfinite(score)) { atomicOr(error, 1U); score = -CUDART_INF_F; }
    if (position >= visible) score = -CUDART_INF_F;
    if (candidates) {
      const unsigned keep = candidates[static_cast<unsigned long long>(token) * positions + position];
      if (keep > 1) atomicOr(error, 1U);
      if (keep != 1) score = -CUDART_INF_F;
    }
    Insert(heap_scores, heap_indices, count, picks, score, position);
  }
  // Sort only the retained indices; output order is position, not score.
  for (unsigned i = 1; i < picks; ++i) {
    const unsigned index = heap_indices[i];
    unsigned j = i;
    while (j && heap_indices[j - 1] > index) { heap_indices[j] = heap_indices[j - 1]; --j; }
    heap_indices[j] = index;
  }
  for (unsigned i = 0; i < picks; ++i) {
    const auto index = heap_indices[i];
    const bool keep = index < visible && (!candidates || candidates[static_cast<unsigned long long>(token) * positions + index] == 1);
    output[static_cast<unsigned long long>(token) * picks + i] = keep ? int(index + offset) : -1;
  }
}
__global__ void Candidates(const __nv_bfloat16* scores, unsigned char* output, unsigned* error,
    unsigned start, unsigned positions) {
  __shared__ float heap_scores[2048];
  __shared__ unsigned heap_indices[2048];
  const unsigned token = blockIdx.x, visible = start + token + 1;
  const unsigned blocks = (positions + 7) / 8, picks = blocks < 2048 ? blocks : 2048;
  const auto row = static_cast<unsigned long long>(token) * positions;
  unsigned count = 0;
  for (unsigned block = 0; block < blocks; ++block) {
    float maximum = -CUDART_INF_F;
    for (unsigned i = 0; i < 8; ++i) {
      const unsigned position = block * 8 + i;
      if (position >= positions) break;
      output[row + position] = 0;
      const float score = __bfloat162float(scores[row + position]);
      if (!isfinite(score)) atomicOr(error, 1U);
      if (position < visible && isfinite(score)) maximum = fmaxf(maximum, score);
    }
    // Pin the block containing the newest causally visible compressed token.
    if (block == (visible - 1) / 8) maximum = CUDART_INF_F;
    Insert(heap_scores, heap_indices, count, picks, maximum, block);
  }
  for (unsigned i = 0; i < picks; ++i) {
    if (heap_scores[i] == -CUDART_INF_F) continue;
    for (unsigned d = 0; d < 8; ++d) {
      const unsigned position = heap_indices[i] * 8 + d;
      if (position < positions) output[row + position] = 1;
    }
  }
}
}
Status LaunchIndexerSelect(const IndexerSelectLaunch& x) {
  const auto validation = ValidateIndexerSelect(x); if (!validation.ok()) return validation;
  if (!x.positions) return Status::Ok();
  Select<<<x.tokens, 1, 0, reinterpret_cast<cudaStream_t>(x.stream)>>>(
      Ptr<const __nv_bfloat16>(x.scores), Ptr<const unsigned char>(x.candidates), Ptr<int>(x.output), Ptr<unsigned>(x.error_flag),
      x.start, x.ratio, x.positions, x.offset);
  const auto error = cudaGetLastError();
  return error == cudaSuccess ? Status::Ok() : Status::Internal(cudaGetErrorString(error));
}
Status LaunchIndexerCandidates(const IndexerCandidatesLaunch& x) {
  const auto validation = ValidateIndexerCandidates(x); if (!validation.ok()) return validation;
  Candidates<<<x.tokens, 1, 0, reinterpret_cast<cudaStream_t>(x.stream)>>>(
      Ptr<const __nv_bfloat16>(x.scores), Ptr<unsigned char>(x.output), Ptr<unsigned>(x.error_flag), x.start, x.positions);
  const auto error = cudaGetLastError();
  return error == cudaSuccess ? Status::Ok() : Status::Internal(cudaGetErrorString(error));
}
}  // namespace pih::deepseek_v41
