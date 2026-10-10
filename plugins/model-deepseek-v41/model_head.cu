#include "model_head.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace pih::deepseek_v41 {
namespace {
template<class T> T* Ptr(EngramDeviceRegion x) { return reinterpret_cast<T*>(x.address); }
__global__ void Project(const __nv_bfloat16* input, const float* weight, float* output, unsigned* error,
    unsigned rows, const unsigned* indices, unsigned input_rows) {
  const unsigned row = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x % 32;
  if (row >= rows) return;
  const unsigned sequence = blockIdx.y;
  const unsigned input_row = indices ? indices[sequence] : 0;
  if (input_row >= input_rows) { atomicOr(error, 4U); if (!lane) output[sequence * rows + row] = 0; return; }
  input += static_cast<unsigned long long>(input_row) * 5120;
  float sum = 0.0f;
  for (unsigned d = lane; d < 5120; d += 32) {
    const float a = __bfloat162float(input[d]), b = weight[static_cast<unsigned long long>(row) * 5120 + d];
    if (!isfinite(a) || !isfinite(b)) atomicOr(error, 1U);
    sum += a * b;
  }
  for (unsigned delta = 16; delta; delta >>= 1) sum += __shfl_down_sync(0xffffffffU, sum, delta);
  if (!lane) {
    if (!isfinite(sum)) { atomicOr(error, 2U); sum = 0.0f; }
    output[sequence * rows + row] = sum;
  }
}
__global__ void TransposeLogits(const float* input, float* output, unsigned sequences, unsigned world) {
  const unsigned at = blockIdx.x * blockDim.x + threadIdx.x;
  if (at >= sequences * 129280) return;
  const unsigned shard = 129280 / world, sequence = at / 129280, token = at % 129280;
  output[at] = input[(static_cast<unsigned long long>(token / shard) * sequences + sequence) * shard + token % shard];
}
}
Status LaunchHeadProjection(const HeadProjectionLaunch& x) {
  const auto validation = ValidateHeadProjection(x); if (!validation.ok()) return validation;
  const unsigned rows = 129280 / x.world_size;
  Project<<<dim3((rows + 3) / 4, x.sequences), 128, 0, reinterpret_cast<cudaStream_t>(x.stream)>>>(Ptr<const __nv_bfloat16>(x.input),
      Ptr<const float>(x.weight), Ptr<float>(x.output), Ptr<unsigned>(x.error_flag), rows,
      x.row_indices.bytes ? Ptr<const unsigned>(x.row_indices) : nullptr, x.input_rows);
  const auto error = cudaGetLastError();
  return error == cudaSuccess ? Status::Ok() : Status::Internal(cudaGetErrorString(error));
}
Status LaunchPackedLogitsTranspose(const PackedLogitsTransposeLaunch& x) {
  const auto valid = ValidatePackedLogitsTranspose(x); if (!valid.ok()) return valid;
  TransposeLogits<<<(x.sequences * 129280 + 255) / 256, 256, 0, reinterpret_cast<cudaStream_t>(x.stream)>>>(
      Ptr<const float>(x.rank_major), Ptr<float>(x.sequence_major), x.sequences, x.world_size);
  const auto error = cudaGetLastError();
  return error == cudaSuccess ? Status::Ok() : Status::Internal(cudaGetErrorString(error));
}
}  // namespace pih::deepseek_v41
