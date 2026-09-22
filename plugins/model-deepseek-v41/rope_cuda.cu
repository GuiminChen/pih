#include "rope_launch.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <numbers>

namespace pih::deepseek_v41 {
namespace {
template<class T> T* Ptr(EngramDeviceRegion region) { return reinterpret_cast<T*>(region.address); }
__global__ void Table(const unsigned* positions, float* phases, unsigned* error,
    bool compressed, float low, float high) {
  const unsigned token = blockIdx.x, pair = threadIdx.x;
  const unsigned position = positions[token];
  if (position >= 1048576) {
    if (!pair) atomicOr(error, 1U);
    phases[token * 64 + pair * 2] = 1.0F;
    phases[token * 64 + pair * 2 + 1] = 0.0F;
    return;
  }
  float frequency = 1.0F / powf(compressed ? 160000.0F : 10000.0F, float(pair * 2) / 64.0F);
  if (compressed) {
    const float ramp = fminf(fmaxf((float(pair) - low) / fmaxf(high - low, 1e-3F), 0.0F), 1.0F);
    const float smooth = 1.0F - ramp;
    frequency = (frequency / 16.0F) * (1.0F - smooth) + frequency * smooth;
  }
  const float phase = float(position) * frequency;
  const float cosine = cosf(phase), sine = sinf(phase);
  if (!isfinite(cosine) || !isfinite(sine)) atomicOr(error, 2U);
  phases[token * 64 + pair * 2] = cosine;
  phases[token * 64 + pair * 2 + 1] = sine;
}
__global__ void Apply(const __nv_bfloat16* input, const float* phases, __nv_bfloat16* output,
    unsigned* error, unsigned heads, unsigned width, bool inverse) {
  const unsigned vector = blockIdx.x, token = vector / heads, lane = threadIdx.x;
  const auto offset = static_cast<unsigned long long>(vector) * width;
  for (unsigned d = lane; d < width - 64; d += 256) output[offset + d] = input[offset + d];
  if (lane >= 32) return;
  const auto element = offset + width - 64 + lane * 2;
  const float real = __bfloat162float(input[element]), imag = __bfloat162float(input[element + 1]);
  const float cosine = phases[token * 64 + lane * 2];
  const float sine = phases[token * 64 + lane * 2 + 1] * (inverse ? -1.0F : 1.0F);
  if (!isfinite(real) || !isfinite(imag) || !isfinite(cosine) || !isfinite(sine)) atomicOr(error, 1U);
  const float rotated_real = real * cosine - imag * sine;
  const float rotated_imag = real * sine + imag * cosine;
  const auto out_real = __float2bfloat16_rn(rotated_real), out_imag = __float2bfloat16_rn(rotated_imag);
  if (!isfinite(rotated_real) || !isfinite(rotated_imag) ||
      !isfinite(__bfloat162float(out_real)) || !isfinite(__bfloat162float(out_imag))) atomicOr(error, 2U);
  output[element] = out_real; output[element + 1] = out_imag;
}
Status LastError() { const auto e = cudaGetLastError(); return e == cudaSuccess ? Status::Ok() : Status::Internal(cudaGetErrorString(e)); }
}
Status LaunchRopeTable(const RopeTableLaunch& x) {
  const auto status = ValidateRopeTable(x); if (!status.ok()) return status;
  const bool compressed = x.layer >= 2 && x.layer < 40;
  // The reference computes correction bounds with host double-precision math.
  const auto correction = [](double rotations) {
    return 64.0 * std::log(65536.0 / (rotations * 2.0 * std::numbers::pi)) / (2.0 * std::log(160000.0));
  };
  const float low = static_cast<float>(std::max(std::floor(correction(32.0)), 0.0));
  const float high = static_cast<float>(std::min(std::ceil(correction(1.0)), 63.0));
  Table<<<x.tokens, 32, 0, reinterpret_cast<cudaStream_t>(x.stream)>>>(Ptr<const unsigned>(x.positions),
      Ptr<float>(x.output), Ptr<unsigned>(x.error_flag), compressed, low, high);
  return LastError();
}
Status LaunchRopeApply(const RopeApplyLaunch& x) {
  const auto status = ValidateRopeApply(x); if (!status.ok()) return status;
  Apply<<<x.tokens * x.heads, 256, 0, reinterpret_cast<cudaStream_t>(x.stream)>>>(Ptr<const __nv_bfloat16>(x.input),
      Ptr<const float>(x.phases), Ptr<__nv_bfloat16>(x.output), Ptr<unsigned>(x.error_flag), x.heads, x.width, x.inverse);
  return LastError();
}
namespace {
__global__ void Sequence(unsigned* positions, unsigned count, unsigned first, unsigned stride) {
  const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) positions[index] = first + index * stride;
}
}
Status LaunchRopeSequence(const RopeSequenceLaunch& x) {
  const auto validation = ValidateRopeSequence(x); if (!validation.ok()) return validation;
  Sequence<<<(x.table.tokens + 255) / 256, 256, 0, reinterpret_cast<cudaStream_t>(x.table.stream)>>>(
      Ptr<unsigned>(x.table.positions), x.table.tokens, x.first_position, x.stride);
  const auto generated = LastError(); if (!generated.ok()) return generated;
  return LaunchRopeTable(x.table);
}
}  // namespace pih::deepseek_v41
