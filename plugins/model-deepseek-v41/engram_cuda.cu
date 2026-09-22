#include "engram_launch.h"
#include <cuda_bf16.h>
#include "linear_fp8.h"
#include <cuda_runtime.h>
#include <math_constants.h>

namespace pih::deepseek_v41 {
namespace {
template<class T> T* Pointer(EngramDeviceRegion region) {
  return reinterpret_cast<T*>(region.address);
}
__device__ float DecodeE4M3(unsigned char bits) {
  const unsigned exponent = (bits >> 3) & 15U, fraction = bits & 7U;
  if (exponent == 15 && fraction == 7) return CUDART_NAN_F;
  const float magnitude = exponent ? ldexpf(1.0F + fraction * 0.125F, int(exponent) - 7)
                                   : ldexpf(float(fraction), -9);
  return bits & 128U ? -magnitude : magnitude;
}
__global__ void Lookup(const unsigned char* table, const unsigned char* scales,
    const unsigned* ids, __nv_bfloat16* output, unsigned* error,
    unsigned global_rows, unsigned partition, unsigned rank) {
  const unsigned hash = blockIdx.x, column = threadIdx.x, id = ids[hash];
  const unsigned first = partition * rank;
  float value = 0.0F;
  if (id >= global_rows) {
    if (!column) atomicOr(error, 1U);
  } else if (id >= first && id - first < partition) {
    const auto row = static_cast<unsigned long long>(id - first);
    const unsigned char scale = scales[row * 8 + column / 32];
    const float decoded = DecodeE4M3(table[row * 256 + column]);
    // E8M0 encodes powers of two with bias 127; 255 is NaN, not infinity.
    value = scale == 255 ? CUDART_NAN_F : decoded * ldexpf(1.0F, int(scale) - 127);
    if (!isfinite(value)) { atomicOr(error, 1U); value = 0.0F; }
  }
  const auto rounded = __float2bfloat16_rn(value);
  if (!isfinite(__bfloat162float(rounded))) atomicOr(error, 2U);
  output[static_cast<unsigned long long>(hash) * 256 + column] = rounded;
}
template<class T> __device__ float LoadGate(const T* data, unsigned index) { return float(data[index]); }
template<> __device__ float LoadGate(const __nv_bfloat16* data, unsigned index) {
  return __bfloat162float(data[index]);
}
template<class T>
__global__ void Gate(const __nv_bfloat16* input, const __nv_bfloat16* kv,
    const T* q, const T* k, const unsigned char* mask, __nv_bfloat16* output, unsigned* error) {
  const unsigned token = blockIdx.x / 4, copy = blockIdx.x % 4;
  const auto input_base = static_cast<unsigned long long>(blockIdx.x) * 5120;
  const auto kv_base = static_cast<unsigned long long>(token) * 25600;
  if (mask && mask[token] != 1) {
    if (mask[token] > 1 && !threadIdx.x) atomicOr(error, 1U);
    for (unsigned d = threadIdx.x; d < 5120; d += blockDim.x) output[input_base + d] = input[input_base + d];
    return;
  }
  __shared__ float sums[3][256];
  float hh = 0, kk = 0, dot = 0;
  for (unsigned d = threadIdx.x; d < 5120; d += blockDim.x) {
    const float h = __bfloat162float(input[input_base + d]);
    const float key = __bfloat162float(kv[kv_base + copy * 5120 + d]);
    const float qw = LoadGate(q, copy * 5120 + d), kw = LoadGate(k, copy * 5120 + d);
    if (!isfinite(h) || !isfinite(key) || !isfinite(qw) || !isfinite(kw)) atomicOr(error, 1U);
    hh += h * h; kk += key * key; dot += (h * (qw * kw)) * key;
  }
  sums[0][threadIdx.x] = hh; sums[1][threadIdx.x] = kk; sums[2][threadIdx.x] = dot;
  __syncthreads();
  for (unsigned stride = 128; stride; stride >>= 1) {
    if (threadIdx.x < stride)
      for (unsigned term = 0; term < 3; ++term) sums[term][threadIdx.x] += sums[term][threadIdx.x + stride];
    __syncthreads();
  }
  if (!threadIdx.x) {
    if (!isfinite(sums[0][0]) || !isfinite(sums[1][0]) || !isfinite(sums[2][0]))
      atomicOr(error, 2U);
    const float normalized = sums[2][0] *
        (rsqrtf(sums[0][0] / 5120.0F + 1e-20F) * rsqrtf(sums[1][0] / 5120.0F + 1e-20F)) *
        rsqrtf(5120.0F);
    const float signed_root = copysignf(sqrtf(fmaxf(fabsf(normalized), 1e-6F)), normalized);
    const float gate = 1.0F / (1.0F + expf(-signed_root));
    if (!isfinite(normalized) || !isfinite(gate)) atomicOr(error, 2U);
    sums[0][0] = gate;
  }
  __syncthreads();
  for (unsigned d = threadIdx.x; d < 5120; d += blockDim.x) {
    const float value = __bfloat162float(kv[kv_base + 20480 + d]);
    const float result = __bfloat162float(input[input_base + d]) + sums[0][0] * value;
    const auto rounded = __float2bfloat16_rn(result);
    if (!isfinite(value) || !isfinite(result) || !isfinite(__bfloat162float(rounded))) atomicOr(error, 2U);
    output[input_base + d] = rounded;
  }
}
Status LaunchStatus() {
  const auto error = cudaGetLastError();
  return error == cudaSuccess ? Status::Ok() : Status::Internal(cudaGetErrorString(error));
}
}
Status LaunchEngramProjection(const EngramProjectionLaunch& x) {
  const auto status = ValidateEngramProjection(x); if (!status.ok()) return status;
  return LaunchFp8Linear({x.input, x.weight, x.weight_scales, x.quantized, x.activation_scales,
      x.output, x.error_flag, x.stream, x.tokens, 6144, 25600});
}
Status LaunchEngramLookup(const EngramLookupLaunch& x) {
  const auto status = ValidateEngramLookup(x);
  if (!status.ok()) return status;
  const unsigned rows = x.layer == 1 ? 384006168U : 384016682U;
  Lookup<<<x.tokens * 24, 256, 0, reinterpret_cast<cudaStream_t>(x.stream)>>>(
      Pointer<const unsigned char>(x.table), Pointer<const unsigned char>(x.scales),
      Pointer<const unsigned>(x.ids), Pointer<__nv_bfloat16>(x.output), Pointer<unsigned>(x.error_flag),
      rows, (rows + x.world_size - 1) / x.world_size, x.rank);
  return LaunchStatus();
}
Status LaunchEngramGate(const EngramGateLaunch& x) {
  const auto status = ValidateEngramGate(x);
  if (!status.ok()) return status;
  const auto stream = reinterpret_cast<cudaStream_t>(x.stream);
  if (x.gate_storage == EngramStorage::kBF16)
    Gate<<<x.tokens * 4, 256, 0, stream>>>(Pointer<const __nv_bfloat16>(x.input),
        Pointer<const __nv_bfloat16>(x.projected_kv), Pointer<const __nv_bfloat16>(x.q_weight),
        Pointer<const __nv_bfloat16>(x.k_weight), Pointer<const unsigned char>(x.mask),
        Pointer<__nv_bfloat16>(x.output), Pointer<unsigned>(x.error_flag));
  else
    Gate<<<x.tokens * 4, 256, 0, stream>>>(Pointer<const __nv_bfloat16>(x.input),
        Pointer<const __nv_bfloat16>(x.projected_kv), Pointer<const float>(x.q_weight),
        Pointer<const float>(x.k_weight), Pointer<const unsigned char>(x.mask),
        Pointer<__nv_bfloat16>(x.output), Pointer<unsigned>(x.error_flag));
  return LaunchStatus();
}
Status LaunchEngramSingleRank(const EngramLaunch& x) {
  const auto validation = ValidateEngramSingleRank(x);
  if (!validation.ok()) return validation;
  const auto lookup = LaunchEngramLookup(x.lookup);
  if (!lookup.ok()) return lookup;
  const auto projection = LaunchEngramProjection(x.projection);
  if (!projection.ok()) return projection;
  return LaunchEngramGate(x.gate);
}
}  // namespace pih::deepseek_v41
