#include <stdint.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "pih/model/qwen3_cuda_invariant.h"

namespace {

using pih::QwenCudaInvariant;

__device__ __forceinline__ float load_bf16(const uint16_t bits) {
  return __uint_as_float(static_cast<uint32_t>(bits) << 16U);
}

__device__ __forceinline__ uint16_t store_bf16(const float value) {
  const uint32_t raw = __float_as_uint(value);
  const uint32_t magnitude = raw & 0x7fffffffU;
  if (magnitude > 0x7f800000U) {
    return static_cast<uint16_t>((raw >> 16U) | 0x0040U);
  }
  const uint32_t rounded = raw + 0x7fffU + ((raw >> 16U) & 1U);
  return static_cast<uint16_t>(rounded >> 16U);
}

__device__ __forceinline__ float load_fp16(const uint16_t bits) {
  return __half2float(__ushort_as_half(bits));
}

__device__ __forceinline__ uint64_t global_thread() {
  return static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
}

__device__ __forceinline__ uint64_t grid_threads() {
  return static_cast<uint64_t>(gridDim.x) * blockDim.x;
}

struct KvHandle final {
  uint32_t slot;
  uint32_t generation;
};

struct KvSlotState final {
  uint32_t generation;
  uint32_t owner_sequence_index;
  uint16_t valid_tokens;
  uint8_t lifecycle;
  uint8_t reserved_zero_u8;
  uint32_t reserved_zero_u32;
};

static_assert(sizeof(KvHandle) == 8);
static_assert(sizeof(KvSlotState) == 16);

constexpr uint32_t kOwnedLifecycle = 3;
constexpr uint32_t kLayers = 28;
constexpr uint32_t kKvHeads = 8;
constexpr uint32_t kQueryHeads = 16;
constexpr uint32_t kHeadDimension = 128;
constexpr uint32_t kTokensPerSlot = 16;
constexpr uint32_t kMaximumBatchTokens = 4096;
constexpr uint32_t kMaximumSequenceTokens = 40960;
constexpr uint32_t kMaximumSlots = 4681;
constexpr uint64_t kElementsPerToken = kKvHeads * kHeadDimension;
constexpr uint64_t kElementsPerPlane = kTokensPerSlot * kElementsPerToken;
constexpr uint64_t kElementsPerLayer = 2 * kElementsPerPlane;
constexpr uint64_t kElementsPerSlot = kLayers * kElementsPerLayer;

__device__ __forceinline__ bool valid_owned_slot(
    const KvHandle handle, const KvSlotState state, const uint32_t owner,
    const uint32_t slot_count) {
  return handle.slot < slot_count && handle.generation != 0 &&
         state.generation == handle.generation &&
         state.owner_sequence_index == owner &&
         state.lifecycle == kOwnedLifecycle && state.reserved_zero_u8 == 0 &&
         state.reserved_zero_u32 == 0;
}

__device__ __forceinline__ uint64_t kv_element_offset(
    const uint32_t slot, const uint32_t layer, const uint32_t plane,
    const uint32_t token, const uint32_t head, const uint32_t column) {
  return static_cast<uint64_t>(slot) * kElementsPerSlot +
         static_cast<uint64_t>(layer) * kElementsPerLayer +
         static_cast<uint64_t>(plane) * kElementsPerPlane +
         static_cast<uint64_t>(token) * kElementsPerToken +
         static_cast<uint64_t>(head) * kHeadDimension + column;
}

__device__ __forceinline__ void publish_error(uint32_t* error_flag,
                                               const QwenCudaInvariant error) {
  if (error_flag != nullptr) {
    atomicCAS(error_flag, 0U, static_cast<uint32_t>(error));
  }
}

}  // namespace

extern "C" __global__ void pih_qwen_w4a16_gemm_compat_v1(
    const uint16_t* input, const uint8_t* packed_weight,
    const uint16_t* scales, uint16_t* output, uint32_t* error_flag,
    uint64_t m, uint64_t n, uint64_t k) {
  const uint64_t output_elements = m * n;
  const uint64_t expected_blocks =
      (output_elements + static_cast<uint64_t>(blockDim.x) - 1U) /
      static_cast<uint64_t>(blockDim.x);
  const bool official_shape =
      (n == 2048 && k == 1024) || (n == 1024 && k == 1024) ||
      (n == 1024 && k == 2048) || (n == 3072 && k == 1024) ||
      (n == 1024 && k == 3072);
  if (error_flag == nullptr || input == nullptr || packed_weight == nullptr ||
      scales == nullptr || output == nullptr || m == 0 || m > 4096 ||
      !official_shape || (k & 127U) != 0 || blockDim.x != 256 ||
      gridDim.x != expected_blocks) {
    publish_error(error_flag, QwenCudaInvariant::kW4A16Launch);
    return;
  }
  for (uint64_t index = global_thread(); index < output_elements;
       index += grid_threads()) {
    const uint64_t row = index / n;
    const uint64_t output_column = index - row * n;
    float accumulator = 0.0F;
    for (uint64_t inner = 0; inner < k; ++inner) {
      const uint8_t packed =
          packed_weight[output_column * (k / 2U) + inner / 2U];
      const uint8_t nibble =
          (inner & 1U) == 0 ? packed & 0x0fU : packed >> 4U;
      if (nibble == 0x08U) {
        publish_error(error_flag, QwenCudaInvariant::kW4A16ReservedNibble);
        return;
      }
      const int32_t signed_value =
          nibble < 8U ? static_cast<int32_t>(nibble)
                      : static_cast<int32_t>(nibble) - 16;
      const float scale =
          load_fp16(scales[output_column * (k / 128U) + inner / 128U]);
      if (!(scale > 0.0F) || !isfinite(scale)) {
        publish_error(error_flag, QwenCudaInvariant::kW4A16InvalidScale);
        return;
      }
      accumulator += load_bf16(input[row * k + inner]) *
                     (static_cast<float>(signed_value) * scale);
    }
    if (!isfinite(accumulator)) {
      publish_error(error_flag, QwenCudaInvariant::kW4A16NonfiniteOutput);
      return;
    }
    output[index] = store_bf16(accumulator);
  }
}

extern "C" __global__ void pih_qwen_teacher_forced_metric_f32_v1(
    const float* logits, const uint32_t* target_tokens,
    uint32_t* argmax_tokens, double* target_nll,
    uint32_t* nonfinite_rows, uint32_t* error_flag,
    uint32_t rows, uint32_t vocabulary_size) {
  if (logits == nullptr || target_tokens == nullptr ||
      argmax_tokens == nullptr || target_nll == nullptr ||
      nonfinite_rows == nullptr || error_flag == nullptr || rows == 0 ||
      rows > 4096 || vocabulary_size != 151936 || blockDim.x != 256 ||
      gridDim.x != rows) {
    publish_error(error_flag, QwenCudaInvariant::kTeacherForcedMetricLaunch);
    return;
  }
  const uint32_t row = blockIdx.x;
  __shared__ float maxima[256];
  __shared__ uint32_t maximum_ids[256];
  __shared__ uint32_t finite_flags[256];
  __shared__ double exponential_sums[256];
  float local_maximum = __int_as_float(static_cast<int>(0xff800000U));
  uint32_t local_id = 0;
  uint32_t local_finite = 1;
  for (uint32_t token = threadIdx.x; token < vocabulary_size;
       token += blockDim.x) {
    const float value = logits[static_cast<uint64_t>(row) * vocabulary_size + token];
    if (!isfinite(value)) local_finite = 0;
    if (value > local_maximum || (value == local_maximum && token < local_id)) {
      local_maximum = value;
      local_id = token;
    }
  }
  maxima[threadIdx.x] = local_maximum;
  maximum_ids[threadIdx.x] = local_id;
  finite_flags[threadIdx.x] = local_finite;
  __syncthreads();
  for (uint32_t stride = 128; stride != 0; stride >>= 1U) {
    if (threadIdx.x < stride) {
      const float candidate = maxima[threadIdx.x + stride];
      const uint32_t candidate_id = maximum_ids[threadIdx.x + stride];
      if (candidate > maxima[threadIdx.x] ||
          (candidate == maxima[threadIdx.x] &&
           candidate_id < maximum_ids[threadIdx.x])) {
        maxima[threadIdx.x] = candidate;
        maximum_ids[threadIdx.x] = candidate_id;
      }
      finite_flags[threadIdx.x] &= finite_flags[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const uint32_t target = target_tokens[row];
  if (threadIdx.x == 0 && target >= vocabulary_size) finite_flags[0] = 0;
  __syncthreads();
  double local_sum = 0.0;
  if (finite_flags[0] != 0) {
    for (uint32_t token = threadIdx.x; token < vocabulary_size;
         token += blockDim.x) {
      local_sum += exp(static_cast<double>(
          logits[static_cast<uint64_t>(row) * vocabulary_size + token] -
          maxima[0]));
    }
  }
  exponential_sums[threadIdx.x] = local_sum;
  __syncthreads();
  for (uint32_t stride = 128; stride != 0; stride >>= 1U) {
    if (threadIdx.x < stride)
      exponential_sums[threadIdx.x] += exponential_sums[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    argmax_tokens[row] = maximum_ids[0];
    nonfinite_rows[row] = finite_flags[0] == 0 ? 1U : 0U;
    target_nll[row] = finite_flags[0] == 0
                          ? 0.0
                          : log(exponential_sums[0]) +
                                static_cast<double>(maxima[0]) -
                                static_cast<double>(
                                    logits[static_cast<uint64_t>(row) *
                                               vocabulary_size + target]);
  }
}

extern "C" __global__ void pih_qwen_embedding_bf16_v1(
    const uint16_t* table, const int64_t* token_ids, uint16_t* output,
    uint64_t token_count, uint64_t vocabulary_size, uint64_t hidden_size) {
  const uint64_t output_elements = token_count * hidden_size;
  for (uint64_t index = global_thread(); index < output_elements;
       index += grid_threads()) {
    const uint64_t token_row = index / hidden_size;
    const int64_t token = token_ids[token_row];
    if (token >= 0 && static_cast<uint64_t>(token) < vocabulary_size) {
      const uint64_t column = index - token_row * hidden_size;
      output[index] =
          table[static_cast<uint64_t>(token) * hidden_size + column];
    }
  }
}

extern "C" __global__ void pih_qwen_embedding_bf16_packed_v2(
    const uint16_t* table, const uint32_t* token_ids, uint16_t* output,
    uint64_t real_token_count, uint64_t vocabulary_size,
    uint64_t hidden_size) {
  const uint64_t output_elements = real_token_count * hidden_size;
  for (uint64_t index = global_thread(); index < output_elements;
       index += grid_threads()) {
    const uint64_t token_row = index / hidden_size;
    const uint32_t token = token_ids[token_row];
    if (static_cast<uint64_t>(token) < vocabulary_size) {
      const uint64_t column = index - token_row * hidden_size;
      output[index] =
          table[static_cast<uint64_t>(token) * hidden_size + column];
    }
  }
}

extern "C" __global__ void pih_qwen_residual_add_bf16_v1(
    const uint16_t* lhs, const uint16_t* rhs, uint16_t* output,
    uint64_t elements) {
  for (uint64_t index = global_thread(); index < elements;
       index += grid_threads()) {
    const float result = load_bf16(lhs[index]) + load_bf16(rhs[index]);
    output[index] = store_bf16(result);
  }
}

extern "C" __global__ void pih_qwen_silu_mul_bf16_v1(
    const uint16_t* gate, const uint16_t* up, uint16_t* output,
    uint64_t elements) {
  for (uint64_t index = global_thread(); index < elements;
       index += grid_threads()) {
    const float gate_value = load_bf16(gate[index]);
    const float silu = gate_value / (1.0F + expf(-gate_value));
    output[index] = store_bf16(silu * load_bf16(up[index]));
  }
}

extern "C" __global__ void pih_qwen_rms_norm_bf16_v1(
    const uint16_t* input, const uint16_t* weight, uint16_t* output,
    uint64_t rows, uint64_t hidden_size, float epsilon) {
  if ((hidden_size != 128 && hidden_size != 1024) || blockDim.x != 256)
    return;
  __shared__ float warp_sums[8];
  for (uint64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    const uint64_t row_offset = row * hidden_size;
    float sum = 0.0F;
    for (uint64_t column = threadIdx.x; column < hidden_size;
         column += blockDim.x) {
      const float value = load_bf16(input[row_offset + column]);
      sum += value * value;
    }
    for (uint32_t delta = 16; delta != 0; delta >>= 1U) {
      sum += __shfl_down_sync(0xffffffffU, sum, delta);
    }
    if ((threadIdx.x & 31U) == 0) warp_sums[threadIdx.x >> 5U] = sum;
    __syncthreads();
    if (threadIdx.x < 32) {
      sum = threadIdx.x < 8 ? warp_sums[threadIdx.x] : 0.0F;
      for (uint32_t delta = 16; delta != 0; delta >>= 1U) {
        sum += __shfl_down_sync(0xffffffffU, sum, delta);
      }
      if (threadIdx.x == 0) {
        warp_sums[0] = 1.0F / sqrtf(sum / static_cast<float>(hidden_size) +
                                    epsilon);
      }
    }
    __syncthreads();
    const float scale = warp_sums[0];
    for (uint64_t column = threadIdx.x; column < hidden_size;
         column += blockDim.x) {
      const float normalized = load_bf16(input[row_offset + column]) * scale;
      output[row_offset + column] =
          store_bf16(normalized * load_bf16(weight[column]));
    }
    __syncthreads();
  }
}

extern "C" __global__ void pih_qwen_rope_bf16_v2(
    const uint16_t* input, const float* cosine, const float* sine,
    uint16_t* output, uint64_t vectors, uint64_t heads, uint64_t head_dim) {
  const uint64_t half = head_dim / 2U;
  const uint64_t pairs = vectors * half;
  for (uint64_t index = global_thread(); index < pairs;
       index += grid_threads()) {
    const uint64_t vector = index / half;
    const uint64_t pair = index - vector * half;
    const uint64_t base = vector * head_dim;
    const float first = load_bf16(input[base + pair]);
    const float second = load_bf16(input[base + half + pair]);
    const uint64_t token = vector / heads;
    const uint64_t angle_index = token * half + pair;
    const float cos_value = cosine[angle_index];
    const float sin_value = sine[angle_index];
    output[base + pair] =
        store_bf16(first * cos_value - second * sin_value);
    output[base + half + pair] =
        store_bf16(second * cos_value + first * sin_value);
  }
}

extern "C" __global__ void pih_qwen_rope_angles_f32_v1(
    const int64_t* positions, float* cosine, float* sine,
    uint64_t token_count, uint64_t head_dim) {
  const uint64_t half = head_dim / 2U;
  const uint64_t values = token_count * half;
  for (uint64_t index = global_thread(); index < values;
       index += grid_threads()) {
    const uint64_t token = index / half;
    const uint64_t pair = index - token * half;
    const float exponent = -2.0F * static_cast<float>(pair) /
                           static_cast<float>(head_dim);
    const float inverse_frequency = powf(1000000.0F, exponent);
    const float angle = static_cast<float>(positions[token]) * inverse_frequency;
    cosine[index] = cosf(angle);
    sine[index] = sinf(angle);
  }
}

extern "C" __global__ void pih_qwen_rope_angles_f32_packed_v2(
    const uint64_t* positions, float* cosine, float* sine,
    uint64_t real_token_count, uint64_t head_dim) {
  const uint64_t half = head_dim / 2U;
  const uint64_t values = real_token_count * half;
  for (uint64_t index = global_thread(); index < values;
       index += grid_threads()) {
    const uint64_t token = index / half;
    const uint64_t pair = index - token * half;
    const float exponent = -2.0F * static_cast<float>(pair) /
                           static_cast<float>(head_dim);
    const float inverse_frequency = powf(1000000.0F, exponent);
    const float angle = static_cast<float>(positions[token]) * inverse_frequency;
    cosine[index] = cosf(angle);
    sine[index] = sinf(angle);
  }
}

extern "C" __global__ void pih_qwen_greedy_argmax_f32_v1(
    const float* logits, int64_t* sampled_token, uint32_t* error_flag,
    uint64_t vocabulary_size) {
  __shared__ float best_values[256];
  __shared__ uint64_t best_tokens[256];
  float best_value = -3.402823466e+38F;
  uint64_t best_token = UINT64_MAX;
  for (uint64_t token = threadIdx.x; token < vocabulary_size;
       token += blockDim.x) {
    const float value = logits[token];
    if (!isfinite(value)) {
      publish_error(error_flag, QwenCudaInvariant::kArgmaxNonfiniteLogit);
      continue;
    }
    if (value > best_value || (value == best_value && token < best_token)) {
      best_value = value;
      best_token = token;
    }
  }
  best_values[threadIdx.x] = best_value;
  best_tokens[threadIdx.x] = best_token;
  __syncthreads();
  for (uint32_t stride = blockDim.x / 2U; stride != 0; stride >>= 1U) {
    if (threadIdx.x < stride) {
      const float other_value = best_values[threadIdx.x + stride];
      const uint64_t other_token = best_tokens[threadIdx.x + stride];
      if (other_value > best_values[threadIdx.x] ||
          (other_value == best_values[threadIdx.x] &&
           other_token < best_tokens[threadIdx.x])) {
        best_values[threadIdx.x] = other_value;
        best_tokens[threadIdx.x] = other_token;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    sampled_token[0] = static_cast<int64_t>(best_tokens[0]);
  }
}

extern "C" __global__ void pih_qwen_kv_append_bf16_v1(
    const uint16_t* key_input, const uint16_t* value_input,
    uint16_t* kv_backing, const KvSlotState* slot_states,
    const KvHandle* handles, const uint16_t* token_offsets,
    uint32_t* error_flag, uint32_t owner_sequence_index, uint32_t layer,
    uint64_t token_count, uint32_t slot_count) {
  if (layer >= kLayers || token_count == 0 ||
      token_count > kMaximumBatchTokens || slot_count == 0 ||
      slot_count > kMaximumSlots) {
    if (global_thread() == 0) {
      publish_error(error_flag, QwenCudaInvariant::kKvAppendLaunch);
    }
    return;
  }
  const uint64_t elements = token_count * 2U * kElementsPerToken;
  for (uint64_t index = global_thread(); index < elements;
       index += grid_threads()) {
    const uint64_t token = index / (2U * kElementsPerToken);
    const uint64_t within_token = index - token * 2U * kElementsPerToken;
    const uint32_t plane = within_token >= kElementsPerToken ? 1U : 0U;
    const uint64_t within_plane = within_token - plane * kElementsPerToken;
    const uint32_t head = static_cast<uint32_t>(within_plane / kHeadDimension);
    const uint32_t column = static_cast<uint32_t>(within_plane % kHeadDimension);
    const KvHandle handle = handles[token];
    if (handle.slot >= slot_count) {
      publish_error(error_flag, QwenCudaInvariant::kKvAppendSlotRange);
      continue;
    }
    const KvSlotState state = slot_states[handle.slot];
    const uint32_t token_offset = token_offsets[token];
    if (!valid_owned_slot(handle, state, owner_sequence_index, slot_count) ||
        token_offset >= kTokensPerSlot) {
      publish_error(error_flag, QwenCudaInvariant::kKvAppendHandleState);
      continue;
    }
    const uint64_t input_index = token * kElementsPerToken + within_plane;
    const uint16_t value = plane == 0 ? key_input[input_index]
                                      : value_input[input_index];
    kv_backing[kv_element_offset(handle.slot, layer, plane, token_offset, head,
                                 column)] = value;
  }
}

extern "C" __global__ void pih_qwen_kv_append_bf16_packed_v2(
    const uint16_t* key_input, const uint16_t* value_input,
    uint16_t* kv_backing, const KvSlotState* slot_states,
    const KvHandle* handles, const uint16_t* token_offsets,
    const uint32_t* request_index, const uint32_t* owner_sequence_indices,
    uint32_t* error_flag, uint32_t layer, uint64_t token_count,
    uint32_t sequence_count, uint32_t slot_count) {
  if (layer >= kLayers || token_count == 0 ||
      token_count > kMaximumBatchTokens || sequence_count == 0 ||
      sequence_count > kMaximumBatchTokens || slot_count == 0 ||
      slot_count > kMaximumSlots) {
    if (global_thread() == 0) {
      publish_error(error_flag, QwenCudaInvariant::kKvAppendLaunch);
    }
    return;
  }
  const uint64_t elements = token_count * 2U * kElementsPerToken;
  for (uint64_t index = global_thread(); index < elements;
       index += grid_threads()) {
    const uint64_t token = index / (2U * kElementsPerToken);
    const uint32_t request = request_index[token];
    if (request >= sequence_count) {
      publish_error(error_flag, QwenCudaInvariant::kKvAppendHandleState);
      continue;
    }
    const uint64_t within_token = index - token * 2U * kElementsPerToken;
    const uint32_t plane = within_token >= kElementsPerToken ? 1U : 0U;
    const uint64_t within_plane = within_token - plane * kElementsPerToken;
    const uint32_t head = static_cast<uint32_t>(within_plane / kHeadDimension);
    const uint32_t column = static_cast<uint32_t>(within_plane % kHeadDimension);
    const KvHandle handle = handles[token];
    if (handle.slot >= slot_count) {
      publish_error(error_flag, QwenCudaInvariant::kKvAppendSlotRange);
      continue;
    }
    const KvSlotState state = slot_states[handle.slot];
    const uint32_t token_offset = token_offsets[token];
    if (!valid_owned_slot(handle, state, owner_sequence_indices[request],
                          slot_count) ||
        token_offset >= kTokensPerSlot) {
      publish_error(error_flag, QwenCudaInvariant::kKvAppendHandleState);
      continue;
    }
    const uint64_t input_index = token * kElementsPerToken + within_plane;
    const uint16_t value = plane == 0 ? key_input[input_index]
                                      : value_input[input_index];
    kv_backing[kv_element_offset(handle.slot, layer, plane, token_offset, head,
                                 column)] = value;
  }
}

extern "C" __global__ void pih_qwen_paged_gqa_bf16_v2(
    const uint16_t* query, uint16_t* output, const uint16_t* kv_backing,
    const KvSlotState* slot_states, const KvHandle* handles,
    uint32_t* error_flag, uint32_t owner_sequence_index, uint32_t layer,
    uint64_t query_start_position, uint32_t query_count,
    uint32_t handle_count, uint32_t key_token_count, float scale,
    uint32_t slot_count) {
  const uint32_t query_row = blockIdx.x / kQueryHeads;
  const uint32_t query_head = blockIdx.x % kQueryHeads;
  if (threadIdx.x != 0 || query_head >= kQueryHeads) return;
  if (layer >= kLayers || query_count == 0 ||
      query_count > kMaximumBatchTokens || query_row >= query_count ||
      handle_count == 0 || key_token_count == 0 ||
      key_token_count > kMaximumSequenceTokens || slot_count == 0 ||
      slot_count > kMaximumSlots ||
      handle_count != (key_token_count + kTokensPerSlot - 1U) /
                          kTokensPerSlot ||
      query_start_position >= key_token_count ||
      query_count > key_token_count - query_start_position ||
      !(scale > 0.0F) || !isfinite(scale)) {
    publish_error(error_flag, QwenCudaInvariant::kPagedGqaLaunch);
    return;
  }

  float accumulator[kHeadDimension];
#pragma unroll
  for (uint32_t column = 0; column < kHeadDimension; ++column) {
    accumulator[column] = 0.0F;
  }
  float running_maximum = -__int_as_float(0x7f800000);
  float running_sum = 0.0F;
  const uint32_t kv_head = query_head / 2U;
  const uint32_t visible_key_count =
      static_cast<uint32_t>(query_start_position) + query_row + 1U;
  for (uint32_t key_token = 0; key_token < visible_key_count; ++key_token) {
    const uint32_t handle_index = key_token / kTokensPerSlot;
    const uint32_t token_offset = key_token % kTokensPerSlot;
    const KvHandle handle = handles[handle_index];
    if (handle.slot >= slot_count) {
      publish_error(error_flag, QwenCudaInvariant::kPagedGqaSlotRange);
      return;
    }
    const KvSlotState state = slot_states[handle.slot];
    if (!valid_owned_slot(handle, state, owner_sequence_index, slot_count) ||
        token_offset >= state.valid_tokens) {
      publish_error(error_flag, QwenCudaInvariant::kPagedGqaHandleState);
      return;
    }
    float score = 0.0F;
#pragma unroll
    for (uint32_t column = 0; column < kHeadDimension; ++column) {
      const float q = load_bf16(
          query[(static_cast<uint64_t>(query_row) * kQueryHeads + query_head) *
                    kHeadDimension +
                column]);
      const float k = load_bf16(kv_backing[kv_element_offset(
          handle.slot, layer, 0, token_offset, kv_head, column)]);
      score += q * k;
    }
    score *= scale;
    if (!isfinite(score)) {
      publish_error(error_flag, QwenCudaInvariant::kPagedGqaNonfiniteScore);
      return;
    }
    const float next_maximum = fmaxf(running_maximum, score);
    const float previous_scale =
        running_sum == 0.0F ? 0.0F : expf(running_maximum - next_maximum);
    const float current_scale = expf(score - next_maximum);
    running_sum = running_sum * previous_scale + current_scale;
#pragma unroll
    for (uint32_t column = 0; column < kHeadDimension; ++column) {
      const float value = load_bf16(kv_backing[kv_element_offset(
          handle.slot, layer, 1, token_offset, kv_head, column)]);
      accumulator[column] = accumulator[column] * previous_scale +
                            current_scale * value;
    }
    running_maximum = next_maximum;
  }
  if (!(running_sum > 0.0F) || !isfinite(running_sum)) {
    publish_error(error_flag,
                  QwenCudaInvariant::kPagedGqaSoftmaxDenominator);
    return;
  }
#pragma unroll
  for (uint32_t column = 0; column < kHeadDimension; ++column) {
    const float normalized = accumulator[column] / running_sum;
    if (!isfinite(normalized)) {
      publish_error(error_flag, QwenCudaInvariant::kPagedGqaNonfiniteOutput);
      return;
    }
    accumulator[column] = normalized;
  }
#pragma unroll
  for (uint32_t column = 0; column < kHeadDimension; ++column) {
    output[(static_cast<uint64_t>(query_row) * kQueryHeads + query_head) *
               kHeadDimension +
           column] =
        store_bf16(accumulator[column]);
  }
}

extern "C" __global__ void pih_qwen_paged_gqa_bf16_packed_v3(
    const uint16_t* query, uint16_t* output, const uint16_t* kv_backing,
    const KvSlotState* slot_states, const KvHandle* handles,
    const uint32_t* visible_handle_offsets, const uint32_t* request_index,
    const uint32_t* query_start_offsets, const uint32_t* key_token_counts,
    const uint32_t* owner_sequence_indices, uint32_t* error_flag,
    uint32_t layer, uint32_t query_count, uint32_t sequence_count, float scale,
    uint32_t slot_count) {
  const uint32_t query_row = blockIdx.x / kQueryHeads;
  const uint32_t query_head = blockIdx.x % kQueryHeads;
  if (threadIdx.x != 0 || query_head >= kQueryHeads) return;
  if (layer >= kLayers || query_count == 0 ||
      query_count > kMaximumBatchTokens || query_row >= query_count ||
      sequence_count == 0 || sequence_count > kMaximumBatchTokens ||
      slot_count == 0 || slot_count > kMaximumSlots || !(scale > 0.0F) ||
      !isfinite(scale)) {
    publish_error(error_flag, QwenCudaInvariant::kPagedGqaLaunch);
    return;
  }
  const uint32_t request = request_index[query_row];
  if (request >= sequence_count) {
    publish_error(error_flag, QwenCudaInvariant::kPagedGqaLaunch);
    return;
  }
  const uint32_t query_begin = query_start_offsets[request];
  const uint32_t query_end = query_start_offsets[request + 1U];
  const uint32_t handle_begin = visible_handle_offsets[request];
  const uint32_t handle_end = visible_handle_offsets[request + 1U];
  const uint32_t key_token_count = key_token_counts[request];
  const uint32_t sequence_query_count = query_end - query_begin;
  const uint32_t handle_count = handle_end - handle_begin;
  if (query_begin > query_row || query_row >= query_end ||
      query_end > query_count || sequence_query_count == 0 ||
      key_token_count == 0 || key_token_count > kMaximumSequenceTokens ||
      sequence_query_count > key_token_count ||
      handle_count != (key_token_count + kTokensPerSlot - 1U) /
                          kTokensPerSlot) {
    publish_error(error_flag, QwenCudaInvariant::kPagedGqaLaunch);
    return;
  }
  const uint32_t local_query_row = query_row - query_begin;
  const uint32_t query_start_position = key_token_count - sequence_query_count;
  const uint32_t visible_key_count =
      query_start_position + local_query_row + 1U;
  const uint32_t owner_sequence_index = owner_sequence_indices[request];

  float accumulator[kHeadDimension];
#pragma unroll
  for (uint32_t column = 0; column < kHeadDimension; ++column) {
    accumulator[column] = 0.0F;
  }
  float running_maximum = -__int_as_float(0x7f800000);
  float running_sum = 0.0F;
  const uint32_t kv_head = query_head / 2U;
  for (uint32_t key_token = 0; key_token < visible_key_count; ++key_token) {
    const KvHandle handle =
        handles[handle_begin + key_token / kTokensPerSlot];
    const uint32_t token_offset = key_token % kTokensPerSlot;
    if (handle.slot >= slot_count) {
      publish_error(error_flag, QwenCudaInvariant::kPagedGqaSlotRange);
      return;
    }
    const KvSlotState state = slot_states[handle.slot];
    if (!valid_owned_slot(handle, state, owner_sequence_index, slot_count) ||
        token_offset >= state.valid_tokens) {
      publish_error(error_flag, QwenCudaInvariant::kPagedGqaHandleState);
      return;
    }
    float score = 0.0F;
#pragma unroll
    for (uint32_t column = 0; column < kHeadDimension; ++column) {
      const float q = load_bf16(
          query[(static_cast<uint64_t>(query_row) * kQueryHeads + query_head) *
                    kHeadDimension +
                column]);
      const float k = load_bf16(kv_backing[kv_element_offset(
          handle.slot, layer, 0, token_offset, kv_head, column)]);
      score += q * k;
    }
    score *= scale;
    if (!isfinite(score)) {
      publish_error(error_flag, QwenCudaInvariant::kPagedGqaNonfiniteScore);
      return;
    }
    const float next_maximum = fmaxf(running_maximum, score);
    const float previous_scale =
        running_sum == 0.0F ? 0.0F : expf(running_maximum - next_maximum);
    const float current_scale = expf(score - next_maximum);
    running_sum = running_sum * previous_scale + current_scale;
#pragma unroll
    for (uint32_t column = 0; column < kHeadDimension; ++column) {
      const float value = load_bf16(kv_backing[kv_element_offset(
          handle.slot, layer, 1, token_offset, kv_head, column)]);
      accumulator[column] = accumulator[column] * previous_scale +
                            current_scale * value;
    }
    running_maximum = next_maximum;
  }
  if (!(running_sum > 0.0F) || !isfinite(running_sum)) {
    publish_error(error_flag,
                  QwenCudaInvariant::kPagedGqaSoftmaxDenominator);
    return;
  }
#pragma unroll
  for (uint32_t column = 0; column < kHeadDimension; ++column) {
    const float normalized = accumulator[column] / running_sum;
    if (!isfinite(normalized)) {
      publish_error(error_flag, QwenCudaInvariant::kPagedGqaNonfiniteOutput);
      return;
    }
    output[(static_cast<uint64_t>(query_row) * kQueryHeads + query_head) *
               kHeadDimension +
           column] = store_bf16(normalized);
  }
}

extern "C" __global__ void pih_qwen_sample_hidden_bf16_packed_v1(
    const uint16_t* input, const uint32_t* sample_rows, uint16_t* output,
    uint32_t* error_flag, uint32_t sample_count, uint32_t packed_token_count,
    uint32_t hidden_size) {
  if (sample_count == 0 || sample_count > packed_token_count ||
      packed_token_count > kMaximumBatchTokens || hidden_size != 1024U) {
    if (global_thread() == 0)
      publish_error(error_flag, QwenCudaInvariant::kPackedSampleLaunch);
    return;
  }
  const uint64_t elements = static_cast<uint64_t>(sample_count) * hidden_size;
  for (uint64_t index = global_thread(); index < elements;
       index += grid_threads()) {
    const uint32_t sample = static_cast<uint32_t>(index / hidden_size);
    const uint32_t column = static_cast<uint32_t>(index % hidden_size);
    const uint32_t source_row = sample_rows[sample];
    if (source_row >= packed_token_count) {
      publish_error(error_flag, QwenCudaInvariant::kPackedSampleRow);
      continue;
    }
    output[index] =
        input[static_cast<uint64_t>(source_row) * hidden_size + column];
  }
}

extern "C" __global__ void pih_qwen_greedy_argmax_f32_packed_v2(
    const float* logits, uint32_t* sampled_token_ids, uint32_t* error_flag,
    uint32_t sample_count, uint32_t vocabulary_size) {
  if (blockDim.x != 256 || gridDim.x != sample_count || sample_count == 0 ||
      sample_count > kMaximumBatchTokens || vocabulary_size != 151936U) {
    if (global_thread() == 0) {
      publish_error(error_flag, QwenCudaInvariant::kPackedArgmaxLaunch);
    }
    return;
  }
  __shared__ float best_values[256];
  __shared__ uint32_t best_tokens[256];
  const uint32_t sample = blockIdx.x;
  float best_value = -3.402823466e+38F;
  uint32_t best_token = UINT32_MAX;
  const uint64_t row_offset =
      static_cast<uint64_t>(sample) * vocabulary_size;
  for (uint32_t token = threadIdx.x; token < vocabulary_size;
       token += blockDim.x) {
    const float value = logits[row_offset + token];
    if (!isfinite(value)) {
      publish_error(error_flag, QwenCudaInvariant::kArgmaxNonfiniteLogit);
      continue;
    }
    if (value > best_value || (value == best_value && token < best_token)) {
      best_value = value;
      best_token = token;
    }
  }
  best_values[threadIdx.x] = best_value;
  best_tokens[threadIdx.x] = best_token;
  __syncthreads();
  for (uint32_t stride = blockDim.x / 2U; stride != 0; stride >>= 1U) {
    if (threadIdx.x < stride) {
      const float other_value = best_values[threadIdx.x + stride];
      const uint32_t other_token = best_tokens[threadIdx.x + stride];
      if (other_value > best_values[threadIdx.x] ||
          (other_value == best_values[threadIdx.x] &&
           other_token < best_tokens[threadIdx.x])) {
        best_values[threadIdx.x] = other_value;
        best_tokens[threadIdx.x] = other_token;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) sampled_token_ids[sample] = best_tokens[0];
}

struct PackedSamplingWire final {
  uint32_t mode;
  float temperature;
  float top_p;
  uint32_t top_k;
  uint64_t seed;
  uint64_t sample_ordinal;
  uint32_t top_logprobs_count;
  uint32_t suppressed_token_ids[17];
  uint32_t suppressed_token_count;
};
static_assert(sizeof(PackedSamplingWire) == 112);

__device__ bool qwen_sampler_suppressed(const PackedSamplingWire& descriptor,
                                        uint32_t token_id) {
  for (uint32_t index = 0; index < descriptor.suppressed_token_count; ++index)
    if (descriptor.suppressed_token_ids[index] == token_id) return true;
  return false;
}

__device__ bool qwen_sampler_suppression_valid(
    const PackedSamplingWire& descriptor, uint32_t vocabulary_size) {
  if (descriptor.suppressed_token_count > 17U ||
      descriptor.suppressed_token_count >= vocabulary_size) return false;
  for (uint32_t index = 0; index < 17U; ++index) {
    if (index >= descriptor.suppressed_token_count) {
      if (descriptor.suppressed_token_ids[index] != 0U) return false;
      continue;
    }
    if (descriptor.suppressed_token_ids[index] >= vocabulary_size) return false;
    for (uint32_t previous = 0; previous < index; ++previous)
      if (descriptor.suppressed_token_ids[previous] ==
          descriptor.suppressed_token_ids[index]) return false;
  }
  return true;
}

__device__ bool qwen_sampler_better(float lv, uint32_t li,
                                    float rv, uint32_t ri) {
  return lv > rv || (lv == rv && li < ri);
}

__device__ void qwen_sampler_swap(float* values, uint32_t* ids,
                                  uint32_t left, uint32_t right) {
  const float value = values[left]; values[left] = values[right];
  values[right] = value;
  const uint32_t id = ids[left]; ids[left] = ids[right]; ids[right] = id;
}

__device__ void qwen_sampler_sift(float* values, uint32_t* ids,
                                  uint32_t root, uint32_t count) {
  while (count > 1U && root <= (count - 2U) / 2U) {
    uint32_t child = root * 2U + 1U;
    if (child + 1U < count &&
        qwen_sampler_better(values[child + 1U], ids[child + 1U],
                            values[child], ids[child])) ++child;
    if (!qwen_sampler_better(values[child], ids[child],
                             values[root], ids[root])) return;
    qwen_sampler_swap(values, ids, root, child);
    root = child;
  }
}

__device__ uint32_t qwen_sampler_philox(uint64_t seed, uint64_t ordinal) {
  const uint64_t block = ordinal / 4U;
  uint32_t counter[4]{static_cast<uint32_t>(block),
                      static_cast<uint32_t>(block >> 32U), 0, 0};
  uint32_t key0 = static_cast<uint32_t>(seed);
  uint32_t key1 = static_cast<uint32_t>(seed >> 32U);
  for (uint32_t round = 0; round < 10; ++round) {
    const uint64_t p0 = static_cast<uint64_t>(0xD2511F53U) * counter[0];
    const uint64_t p1 = static_cast<uint64_t>(0xCD9E8D57U) * counter[2];
    const uint32_t next[4]{
        static_cast<uint32_t>(p1 >> 32U) ^ counter[1] ^ key0,
        static_cast<uint32_t>(p1),
        static_cast<uint32_t>(p0 >> 32U) ^ counter[3] ^ key1,
        static_cast<uint32_t>(p0)};
    for (uint32_t lane = 0; lane < 4; ++lane) counter[lane] = next[lane];
    key0 += 0x9E3779B9U; key1 += 0xBB67AE85U;
  }
  return counter[ordinal % 4U];
}

// Correctness-first pih-sampler-v1 implementation. One thread owns one
// sample so ordering, rounding, tie-breaks, and Philox consumption are frozen.
extern "C" __global__ void pih_qwen_sample_f32_packed_v1(
    float* logits, const PackedSamplingWire* descriptors,
    const uint32_t* sample_sequence_indices, uint32_t* workspace_ids,
    uint32_t* sampled_token_ids, float* selected_logprobs,
    uint32_t* rng_words, uint32_t* top_token_ids, float* top_logprobs,
    uint32_t* top_counts, uint32_t* error_flag, uint32_t sample_count,
    uint32_t sequence_count, uint32_t vocabulary_size) {
  if (blockDim.x != 1 || gridDim.x != sample_count || threadIdx.x != 0 ||
      sample_count == 0 || sample_count > kMaximumBatchTokens ||
      sequence_count == 0 || sequence_count > kMaximumBatchTokens ||
      vocabulary_size != 151936U) {
    if (global_thread() == 0)
      publish_error(error_flag, QwenCudaInvariant::kPackedSamplerLaunch);
    return;
  }
  const uint32_t sample = blockIdx.x;
  const uint32_t sequence = sample_sequence_indices[sample];
  if (sequence >= sequence_count) {
    publish_error(error_flag, QwenCudaInvariant::kPackedSamplerDescriptor);
    return;
  }
  const PackedSamplingWire d = descriptors[sequence];
  const bool stochastic = d.mode == 1U;
  if ((d.mode > 1U) || !isfinite(d.top_p) || d.top_p <= 0.0F ||
      d.top_p > 1.0F || d.top_logprobs_count > 20U ||
      !qwen_sampler_suppression_valid(d, vocabulary_size) ||
      d.top_logprobs_count > vocabulary_size - d.suppressed_token_count ||
      (!stochastic && (d.temperature != 0.0F || d.top_p != 1.0F ||
                       d.top_k != 0U)) ||
      (stochastic && (!isfinite(d.temperature) || d.temperature <= 0.0F ||
                      d.temperature > 2.0F || d.top_k > vocabulary_size))) {
    publish_error(error_flag, QwenCudaInvariant::kPackedSamplerDescriptor);
    return;
  }
  float* values = logits + static_cast<uint64_t>(sample) * vocabulary_size;
  uint32_t* ids = workspace_ids +
                  static_cast<uint64_t>(sample) * vocabulary_size;
  const float divisor = stochastic ? d.temperature : 1.0F;
  float maximum = -3.402823466e+38F;
  for (uint32_t id = 0; id < vocabulary_size; ++id) {
    const float raw = values[id];
    if (!isfinite(raw)) {
      publish_error(error_flag, QwenCudaInvariant::kArgmaxNonfiniteLogit);
      return;
    }
    const float scaled = qwen_sampler_suppressed(d, id)
                             ? __int_as_float(static_cast<int>(0xFF800000U))
                             : raw / divisor;
    values[id] = scaled; ids[id] = id; maximum = fmaxf(maximum, scaled);
  }
  float full_sum = 0.0F;
  for (uint32_t id = 0; id < vocabulary_size; ++id)
    full_sum += expf(values[id] - maximum);
  if (!isfinite(full_sum) || full_sum <= 0.0F) {
    publish_error(error_flag, QwenCudaInvariant::kPackedSamplerDescriptor);
    return;
  }
  const float logsumexp = maximum + logf(full_sum);
  for (int64_t root = static_cast<int64_t>(vocabulary_size / 2U) - 1;
       root >= 0; --root)
    qwen_sampler_sift(values, ids, static_cast<uint32_t>(root), vocabulary_size);
  for (uint32_t count = vocabulary_size; count > 1U; --count) {
    qwen_sampler_swap(values, ids, 0, count - 1U);
    qwen_sampler_sift(values, ids, 0, count - 1U);
  }
  for (uint32_t left = 0, right = vocabulary_size - 1U; left < right;
       ++left, --right) qwen_sampler_swap(values, ids, left, right);
  top_counts[sample] = d.top_logprobs_count;
  for (uint32_t rank = 0; rank < d.top_logprobs_count; ++rank) {
    const uint64_t output = static_cast<uint64_t>(sample) * 20U + rank;
    top_token_ids[output] = ids[rank];
    top_logprobs[output] = values[rank] - logsumexp;
  }
  if (!stochastic) {
    sampled_token_ids[sample] = ids[0];
    selected_logprobs[sample] = values[0] - logsumexp;
    rng_words[sample] = 0;
    return;
  }
  const uint32_t allowed = vocabulary_size - d.suppressed_token_count;
  const uint32_t candidates = d.top_k == 0U || d.top_k > allowed
                                  ? allowed : d.top_k;
  const float filtered_max = values[0];
  float candidate_sum = 0.0F;
  for (uint32_t index = 0; index < candidates; ++index) {
    values[index] = expf(values[index] - filtered_max);
    candidate_sum += values[index];
  }
  float prefix = 0.0F; uint32_t retained = 0;
  do { prefix += values[retained++]; }
  while (retained < candidates && prefix / candidate_sum < d.top_p);
  float retained_sum = 0.0F;
  for (uint32_t index = 0; index < retained; ++index)
    retained_sum += values[index];
  const uint32_t word = qwen_sampler_philox(d.seed, d.sample_ordinal);
  const double uniform = (static_cast<double>(word) + 0.5) / 4294967296.0;
  double cumulative = 0.0; uint32_t selected = retained - 1U;
  for (uint32_t index = 0; index < retained; ++index) {
    cumulative += static_cast<double>(values[index] / retained_sum);
    if (cumulative > uniform) { selected = index; break; }
  }
  sampled_token_ids[sample] = ids[selected];
  selected_logprobs[sample] =
      logf(values[selected]) + filtered_max - logsumexp;
  rng_words[sample] = word;
}
