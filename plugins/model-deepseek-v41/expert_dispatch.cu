#include "expert_dispatch.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace pih::deepseek_v41 {
namespace {
template<class T> T* Ptr(EngramDeviceRegion x) { return reinterpret_cast<T*>(x.address); }
__global__ void Dispatch(const unsigned* indices, unsigned* counts, unsigned* slots, unsigned* error,
    unsigned tokens, unsigned experts, unsigned picks, unsigned first_expert) {
  const unsigned local = blockIdx.x, expert = first_expert + local;
  const auto row = static_cast<unsigned long long>(local) * tokens;
  unsigned count = 0;
  for (unsigned token = 0; token < tokens; ++token) {
    unsigned found = 0xffffffffU;
    for (unsigned pick = 0; pick < picks; ++pick) {
      const unsigned slot = token * picks + pick, id = indices[slot];
      if (id >= experts) atomicOr(error, 1U);
      // Validate uniqueness even when a malformed duplicate refers to a remote
      // expert, so every rank rejects the same invalid global route table.
      for (unsigned prior = 0; prior < pick; ++prior)
        if (indices[token * picks + prior] == id) atomicOr(error, 1U);
      if (id == expert && found == 0xffffffffU) found = slot;
    }
    if (found != 0xffffffffU) slots[row + count++] = found;
  }
  for (unsigned i = count; i < tokens; ++i) slots[row + i] = 0xffffffffU;
  counts[local] = count;
}
}
Status LaunchExpertDispatch(const ExpertDispatchLaunch& x) {
  const auto validation = ValidateExpertDispatch(x); if (!validation.ok()) return validation;
  const unsigned experts = x.layer < 40 ? 384 : 128, picks = x.layer < 40 ? 6 : 3;
  const unsigned local = experts / x.world_size;
  Dispatch<<<local, 1, 0, reinterpret_cast<cudaStream_t>(x.stream)>>>(Ptr<const unsigned>(x.indices),
      Ptr<unsigned>(x.counts), Ptr<unsigned>(x.slots), Ptr<unsigned>(x.error_flag), x.tokens, experts, picks, x.rank * local);
  const auto error = cudaGetLastError();
  return error == cudaSuccess ? Status::Ok() : Status::Internal(cudaGetErrorString(error));
}
namespace {
__global__ void Gather(const unsigned* indices, const unsigned* counts, const unsigned* slots,
    const __nv_bfloat16* input, const float* weights, __nv_bfloat16* output, float* gathered_weights,
    unsigned* error, unsigned tokens, unsigned picks, unsigned local, unsigned expert, unsigned rows) {
  const unsigned row = blockIdx.x;
  const bool count_valid = counts[local] == rows;
  if (!count_valid && !threadIdx.x) atomicOr(error, 1U);
  if (!rows) return;
  const unsigned slot = slots[static_cast<unsigned long long>(local) * tokens + row];
  bool valid = count_valid && slot < tokens * picks;
  if (valid) valid = indices[slot] == expert;
  if (valid && row) {
    const unsigned previous = slots[static_cast<unsigned long long>(local) * tokens + row - 1];
    valid = previous < tokens * picks && previous / picks < slot / picks;
  }
  const float weight = valid ? weights[slot] : 0.0f;
  if (!isfinite(weight) || weight < 0.0f) valid = false;
  if (!threadIdx.x) {
    if (!valid) atomicOr(error, 1U);
    gathered_weights[row] = valid ? weight : 0.0f;
  }
  for (unsigned column = threadIdx.x; column < 5120; column += 256)
    output[static_cast<unsigned long long>(row) * 5120 + column] = valid
        ? input[static_cast<unsigned long long>(slot / picks) * 5120 + column] : __float2bfloat16_rn(0.0f);
}
}
Status LaunchExpertGather(const ExpertGatherLaunch& x) {
  const auto validation = ValidateExpertGather(x); if (!validation.ok()) return validation;
  const auto& d = x.dispatch;
  const unsigned experts = d.layer < 40 ? 384 : 128, picks = d.layer < 40 ? 6 : 3;
  const unsigned local = x.expert - d.rank * (experts / d.world_size);
  Gather<<<x.rows ? x.rows : 1, 256, 0, reinterpret_cast<cudaStream_t>(d.stream)>>>(
      Ptr<const unsigned>(d.indices), Ptr<const unsigned>(d.counts), Ptr<const unsigned>(d.slots),
      Ptr<const __nv_bfloat16>(x.input), Ptr<const float>(x.route_weights), Ptr<__nv_bfloat16>(x.output),
      Ptr<float>(x.gathered_weights), Ptr<unsigned>(d.error_flag), d.tokens, picks, local, x.expert, x.rows);
  const auto error = cudaGetLastError();
  return error == cudaSuccess ? Status::Ok() : Status::Internal(cudaGetErrorString(error));
}
namespace {
__global__ void Scatter(const unsigned* indices, const unsigned* counts, const unsigned* slots,
    const __nv_bfloat16* input, float* accumulator, unsigned* error,
    unsigned tokens, unsigned picks, unsigned local, unsigned expert, unsigned rows) {
  const unsigned row = blockIdx.x;
  const bool count_valid = counts[local] == rows;
  if (!count_valid && !threadIdx.x) atomicOr(error, 1U);
  if (!rows) return;
  const unsigned slot = slots[static_cast<unsigned long long>(local) * tokens + row];
  bool valid = count_valid && slot < tokens * picks;
  if (valid) valid = indices[slot] == expert;
  if (valid && row) {
    const unsigned previous = slots[static_cast<unsigned long long>(local) * tokens + row - 1];
    valid = previous < tokens * picks && previous / picks < slot / picks;
  }
  if (!valid) { if (!threadIdx.x) atomicOr(error, 1U); return; }
  for (unsigned column = threadIdx.x; column < 5120; column += 256) {
    const float value = __bfloat162float(input[static_cast<unsigned long long>(row) * 5120 + column]);
    if (!isfinite(value)) { atomicOr(error, 2U); continue; }
    // Atomics also avoid a data race for malformed non-adjacent duplicate slots.
    // A valid dispatch has one row per token, hence no same-expert contention.
    const float previous = atomicAdd(accumulator + static_cast<unsigned long long>(slot / picks) * 5120 + column, value);
    if (!isfinite(previous) || !isfinite(previous + value)) atomicOr(error, 2U);
  }
}
}
Status LaunchExpertScatter(const ExpertScatterLaunch& x) {
  const auto validation = ValidateExpertScatter(x); if (!validation.ok()) return validation;
  const auto& d = x.dispatch;
  const unsigned experts = d.layer < 40 ? 384 : 128, picks = d.layer < 40 ? 6 : 3;
  const unsigned local = x.expert - d.rank * (experts / d.world_size);
  Scatter<<<x.rows ? x.rows : 1, 256, 0, reinterpret_cast<cudaStream_t>(d.stream)>>>(
      Ptr<const unsigned>(d.indices), Ptr<const unsigned>(d.counts), Ptr<const unsigned>(d.slots),
      Ptr<const __nv_bfloat16>(x.input), Ptr<float>(x.accumulator), Ptr<unsigned>(d.error_flag),
      d.tokens, picks, local, x.expert, x.rows);
  const auto error = cudaGetLastError();
  return error == cudaSuccess ? Status::Ok() : Status::Internal(cudaGetErrorString(error));
}
}  // namespace pih::deepseek_v41
