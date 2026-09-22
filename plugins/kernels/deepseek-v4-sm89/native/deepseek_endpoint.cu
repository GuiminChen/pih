#include "pih/backend/cuda/deepseek_endpoint.h"

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cfloat>
#include <climits>

#include "pih/backend/cuda/cuda_status.h"

namespace pih { namespace {

__device__ float reduce_sum(float value, float* scratch) {
  scratch[threadIdx.x] = value;
  __syncthreads();
  for (std::uint32_t width = blockDim.x / 2; width != 0; width >>= 1U) {
    if (threadIdx.x < width) scratch[threadIdx.x] += scratch[threadIdx.x + width];
    __syncthreads();
  }
  const auto result = scratch[0];
  // All warps must read the result before a subsequent reduction reuses scratch.
  __syncthreads();
  return result;
}

__device__ float sigmoid(float value) {
  if (value >= 0.0F) return 1.0F / (1.0F + expf(-value));
  const auto e = expf(value);
  return e / (1.0F + e);
}

__global__ void embedding_kernel(
    const std::uint32_t* token_ids, const __nv_bfloat16* weight,
    __nv_bfloat16* output, std::uint32_t* error, std::uint64_t elements,
    std::uint32_t hidden_size, std::uint32_t vocab_size) {
  const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
  if (index >= elements) return;
  const auto column = index % hidden_size;
  const auto token = index / (4U * hidden_size);
  const auto id = token_ids[token];
  if (id >= vocab_size) {
    atomicOr(error, 1U);
    output[index] = __float2bfloat16_rn(0.0F);
    return;
  }
  const auto value = weight[static_cast<std::uint64_t>(id) * hidden_size + column];
  if (!isfinite(__bfloat162float(value))) atomicOr(error, 2U);
  output[index] = value;
}

__global__ void hc_head_kernel(
    const __nv_bfloat16* input, const float* fn, const float* scale,
    const float* base, __nv_bfloat16* output, std::uint32_t* error,
    std::uint32_t hidden_size, float rms_epsilon, float hc_epsilon) {
  __shared__ float scratch[256];
  __shared__ float gates[4];
  const auto width = 4U * hidden_size;
  const auto input_base = static_cast<std::uint64_t>(blockIdx.x) * width;
  float squares = 0.0F;
  for (std::uint32_t column = threadIdx.x; column < width; column += blockDim.x) {
    const auto value = __bfloat162float(input[input_base + column]);
    if (!isfinite(value)) atomicOr(error, 4U);
    squares += value * value;
  }
  const auto inverse = rsqrtf(reduce_sum(squares, scratch) / width + rms_epsilon);
  for (std::uint32_t stream = 0; stream < 4; ++stream) {
    float partial = 0.0F;
    for (std::uint32_t column = threadIdx.x; column < width;
         column += blockDim.x) {
      const auto value = fn[static_cast<std::uint64_t>(stream) * width + column];
      if (!isfinite(value)) atomicOr(error, 8U);
      partial += __bfloat162float(input[input_base + column]) * value;
    }
    const auto dot = reduce_sum(partial, scratch);
    if (threadIdx.x == 0) {
      if (!isfinite(scale[0]) || !isfinite(base[stream])) atomicOr(error, 16U);
      gates[stream] = sigmoid(dot * inverse * scale[0] + base[stream]) + hc_epsilon;
    }
    __syncthreads();
  }
  const auto output_base = static_cast<std::uint64_t>(blockIdx.x) * hidden_size;
  for (std::uint32_t column = threadIdx.x; column < hidden_size;
       column += blockDim.x) {
    float sum = 0.0F;
    for (std::uint32_t stream = 0; stream < 4; ++stream)
      sum += gates[stream] * __bfloat162float(input[
          input_base + static_cast<std::uint64_t>(stream) * hidden_size + column]);
    if (!isfinite(sum)) atomicOr(error, 32U);
    output[output_base + column] = __float2bfloat16_rn(sum);
  }
}

__global__ void lm_head_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* weight, float* logits,
    std::uint32_t* error, std::uint32_t hidden_size,
    std::uint32_t vocab_size) {
  __shared__ float scratch[256];
  const auto row = blockIdx.y;
  const auto word = blockIdx.x;
  float partial = 0.0F;
  for (std::uint32_t column = threadIdx.x; column < hidden_size;
       column += blockDim.x) {
    const auto x = __bfloat162float(input[
        static_cast<std::uint64_t>(row) * hidden_size + column]);
    const auto w = __bfloat162float(weight[
        static_cast<std::uint64_t>(word) * hidden_size + column]);
    if (!isfinite(x) || !isfinite(w)) atomicOr(error, 64U);
    partial += x * w;
  }
  const auto sum = reduce_sum(partial, scratch);
  if (threadIdx.x == 0) {
    if (!isfinite(sum)) atomicOr(error, 128U);
    logits[static_cast<std::uint64_t>(row) * vocab_size + word] = sum;
  }
}

#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
__global__ void dspark_markov_kernel(
    const std::uint32_t* ids, const __nv_bfloat16* embedding,
    const __nv_bfloat16* head, const float* raw,
    __nv_bfloat16* output_embedding, float* biased, std::uint32_t* error,
    std::uint32_t vocab_size, std::uint32_t rank) {
  __shared__ float scratch[256];
  const auto row = blockIdx.y;
  const auto word = blockIdx.x;
  const auto id = ids[row];
  if (id >= vocab_size) {
    if (threadIdx.x == 0) atomicOr(error, 256U);
    return;
  }
  float partial = 0.0F;
  for (std::uint32_t column = threadIdx.x; column < rank;
       column += blockDim.x) {
    const auto value = embedding[static_cast<std::uint64_t>(id) * rank + column];
    if (word == 0) output_embedding[static_cast<std::uint64_t>(row) * rank +
                                    column] = value;
    const auto x = __bfloat162float(value);
    const auto w = __bfloat162float(
        head[static_cast<std::uint64_t>(word) * rank + column]);
    if (!isfinite(x) || !isfinite(w)) atomicOr(error, 512U);
    partial += x * w;
  }
  const auto sum = reduce_sum(partial, scratch);
  if (threadIdx.x == 0) {
    const auto base = raw[static_cast<std::uint64_t>(row) * vocab_size + word];
    const auto value = base + sum;
    if (!isfinite(base) || !isfinite(value)) atomicOr(error, 1024U);
    biased[static_cast<std::uint64_t>(row) * vocab_size + word] = value;
  }
}

__global__ void dspark_confidence_kernel(
    const __nv_bfloat16* hidden, const __nv_bfloat16* markov,
    const __nv_bfloat16* weight, float* output, std::uint32_t* error,
    std::uint32_t hidden_size, std::uint32_t rank) {
  __shared__ float scratch[256];
  const auto row = blockIdx.x;
  float partial = 0.0F;
  for (std::uint32_t column = threadIdx.x; column < hidden_size + rank;
       column += blockDim.x) {
    const auto x = column < hidden_size
        ? __bfloat162float(hidden[static_cast<std::uint64_t>(row) * hidden_size +
                                  column])
        : __bfloat162float(markov[static_cast<std::uint64_t>(row) * rank +
                                  column - hidden_size]);
    const auto w = __bfloat162float(weight[column]);
    if (!isfinite(x) || !isfinite(w)) atomicOr(error, 2048U);
    partial += x * w;
  }
  const auto sum = reduce_sum(partial, scratch);
  if (threadIdx.x == 0) {
    if (!isfinite(sum)) atomicOr(error, 4096U);
    output[row] = sum;
  }
}
#endif

__device__ bool deepseek_token_suppressed(
    const DeepSeekSuppressedTokenSet& suppressed,
    std::uint32_t token_id) {
  for (std::uint32_t index = 0; index < suppressed.token_count; ++index)
    if (suppressed.token_ids[index] == token_id) return true;
  return false;
}

__global__ void argmax_kernel(const float* logits,
                                     std::uint32_t* token_id,
                                     float* selected_logprob,
                                     std::uint32_t* top_ids,
                                     float* top_logprobs,
                                     std::uint32_t top_count,
                                     DeepSeekSuppressedTokenSet suppressed,
                                     std::uint32_t* error,
                                     std::uint32_t vocab_size) {
  __shared__ float values[256];
  __shared__ std::uint32_t ids[256];
  float best_value = -FLT_MAX;
  std::uint32_t best_id = UINT_MAX;
  for (std::uint32_t id = threadIdx.x; id < vocab_size; id += blockDim.x) {
    const auto value = logits[id];
    if (!isfinite(value)) atomicOr(error, 8192U);
    if (deepseek_token_suppressed(suppressed, id)) continue;
    if (value > best_value || (value == best_value && id < best_id)) {
      best_value = value;
      best_id = id;
    }
  }
  values[threadIdx.x] = best_value;
  ids[threadIdx.x] = best_id;
  __syncthreads();
  for (std::uint32_t width = blockDim.x / 2; width != 0; width >>= 1U) {
    if (threadIdx.x < width) {
      const auto other_value = values[threadIdx.x + width];
      const auto other_id = ids[threadIdx.x + width];
      if (other_value > values[threadIdx.x] ||
          (other_value == values[threadIdx.x] && other_id < ids[threadIdx.x])) {
        values[threadIdx.x] = other_value;
        ids[threadIdx.x] = other_id;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    token_id[0] = ids[0];
    if (selected_logprob != nullptr) {
      const auto maximum = values[0];
      float sum = 0.0F;
      for (std::uint32_t id = 0; id < vocab_size; ++id) {
        if (deepseek_token_suppressed(suppressed, id)) continue;
        sum += expf(logits[id] - maximum);
      }
      if (!isfinite(sum) || sum <= 0.0F) {
        atomicOr(error, 131072U);
        return;
      }
      const auto logsumexp = maximum + logf(sum);
      selected_logprob[0] = logits[ids[0]] - logsumexp;
      float ranked_values[20];
      std::uint32_t ranked_ids[20];
      for (std::uint32_t index = 0; index < top_count; ++index) {
        ranked_values[index] = -FLT_MAX;
        ranked_ids[index] = UINT_MAX;
      }
      for (std::uint32_t id = 0; id < vocab_size; ++id) {
        if (deepseek_token_suppressed(suppressed, id)) continue;
        std::uint32_t position = top_count;
        for (std::uint32_t index = 0; index < top_count; ++index) {
          if (logits[id] > ranked_values[index] ||
              (logits[id] == ranked_values[index] && id < ranked_ids[index])) {
            position = index;
            break;
          }
        }
        if (position == top_count) continue;
        for (std::uint32_t index = top_count - 1U; index > position; --index) {
          ranked_values[index] = ranked_values[index - 1U];
          ranked_ids[index] = ranked_ids[index - 1U];
        }
        ranked_values[position] = logits[id];
        ranked_ids[position] = id;
      }
      for (std::uint32_t index = 0; index < top_count; ++index) {
        top_ids[index] = ranked_ids[index];
        top_logprobs[index] = ranked_values[index] - logsumexp;
      }
    }
  }
}

__device__ bool sampler_better(float left_value, std::uint32_t left_id,
                               float right_value, std::uint32_t right_id) {
  return left_value > right_value ||
         (left_value == right_value && left_id < right_id);
}

__device__ void sampler_swap(float* values, std::uint32_t* ids,
                             std::uint32_t left, std::uint32_t right) {
  const auto value = values[left];
  values[left] = values[right];
  values[right] = value;
  const auto id = ids[left];
  ids[left] = ids[right];
  ids[right] = id;
}

__device__ void sampler_sift_down(float* values, std::uint32_t* ids,
                                  std::uint32_t root,
                                  std::uint32_t count) {
  while (root <= (count - 2U) / 2U) {
    auto child = root * 2U + 1U;
    if (child + 1U < count &&
        sampler_better(values[child + 1U], ids[child + 1U],
                       values[child], ids[child])) {
      ++child;
    }
    if (!sampler_better(values[child], ids[child],
                        values[root], ids[root])) {
      return;
    }
    sampler_swap(values, ids, root, child);
    root = child;
  }
}

__device__ std::uint32_t sampler_philox_word(std::uint64_t seed,
                                             std::uint64_t ordinal) {
  const auto block = ordinal / 4U;
  std::uint32_t counter[4]{static_cast<std::uint32_t>(block),
                           static_cast<std::uint32_t>(block >> 32U), 0, 0};
  auto key0 = static_cast<std::uint32_t>(seed);
  auto key1 = static_cast<std::uint32_t>(seed >> 32U);
  for (std::uint32_t round = 0; round < 10; ++round) {
    const auto product0 = static_cast<std::uint64_t>(0xD2511F53U) * counter[0];
    const auto product1 = static_cast<std::uint64_t>(0xCD9E8D57U) * counter[2];
    const std::uint32_t next[4]{
        static_cast<std::uint32_t>(product1 >> 32U) ^ counter[1] ^ key0,
        static_cast<std::uint32_t>(product1),
        static_cast<std::uint32_t>(product0 >> 32U) ^ counter[3] ^ key1,
        static_cast<std::uint32_t>(product0)};
    for (std::uint32_t lane = 0; lane < 4; ++lane) counter[lane] = next[lane];
    key0 += 0x9E3779B9U;
    key1 += 0xBB67AE85U;
  }
  return counter[ordinal % 4U];
}

// Correctness-first pih-sampler-v1 compatibility kernel. It deliberately
// uses one CUDA thread so the reduction and ordering tree are fully frozen.
// Optimized dispatches must prove bit-exact equivalence before replacing it.
__global__ void stochastic_sample_compatibility_kernel(
    const float* logits, std::uint32_t* token_id, float* selected_logprob,
    std::uint32_t* rng_word, float* values, std::uint32_t* ids,
    std::uint32_t* error, std::uint32_t vocab_size, float temperature,
    float top_p, std::uint32_t top_k, std::uint64_t seed,
    std::uint64_t ordinal, std::uint32_t* top_ids, float* top_logprobs,
    std::uint32_t top_count, DeepSeekSuppressedTokenSet suppressed) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  float maximum = -FLT_MAX;
  for (std::uint32_t id = 0; id < vocab_size; ++id) {
    const auto raw = logits[id];
    if (!isfinite(raw)) {
      atomicOr(error, 65536U);
      return;
    }
    const auto scaled = deepseek_token_suppressed(suppressed, id)
        ? __int_as_float(static_cast<int>(0xFF800000U))
                                             : raw / temperature;
    values[id] = scaled;
    ids[id] = id;
    maximum = fmaxf(maximum, scaled);
  }
  float full_sum = 0.0F;
  for (std::uint32_t id = 0; id < vocab_size; ++id) {
    full_sum += expf(values[id] - maximum);
  }
  if (!isfinite(full_sum) || full_sum <= 0.0F) {
    atomicOr(error, 131072U);
    return;
  }
  const auto logsumexp = maximum + logf(full_sum);

  for (std::int64_t root = static_cast<std::int64_t>(vocab_size / 2U) - 1;
       root >= 0; --root) {
    sampler_sift_down(values, ids, static_cast<std::uint32_t>(root),
                      vocab_size);
  }
  for (auto count = vocab_size; count > 1U; --count) {
    sampler_swap(values, ids, 0, count - 1U);
    sampler_sift_down(values, ids, 0, count - 1U);
  }
  for (std::uint32_t left = 0, right = vocab_size - 1U; left < right;
       ++left, --right) {
    sampler_swap(values, ids, left, right);
  }
  for (std::uint32_t index = 0; index < top_count; ++index) {
    top_ids[index] = ids[index];
    top_logprobs[index] = values[index] - logsumexp;
  }

  const auto allowed_count = vocab_size - suppressed.token_count;
  const auto candidates = top_k == 0 || top_k > allowed_count
      ? allowed_count : top_k;
  const auto filtered_maximum = values[0];
  float candidate_sum = 0.0F;
  for (std::uint32_t index = 0; index < candidates; ++index) {
    values[index] = expf(values[index] - filtered_maximum);
    candidate_sum += values[index];
  }
  float prefix = 0.0F;
  std::uint32_t retained = 0;
  do {
    prefix += values[retained++];
  } while (retained < candidates && prefix / candidate_sum < top_p);
  float retained_sum = 0.0F;
  for (std::uint32_t index = 0; index < retained; ++index) {
    retained_sum += values[index];
  }
  const auto word = sampler_philox_word(seed, ordinal);
  const auto uniform = (static_cast<double>(word) + 0.5) / 4294967296.0;
  double cumulative = 0.0;
  auto selected = retained - 1U;
  for (std::uint32_t index = 0; index < retained; ++index) {
    cumulative += static_cast<double>(values[index] / retained_sum);
    if (cumulative > uniform) {
      selected = index;
      break;
    }
  }
  const auto selected_id = ids[selected];
  token_id[0] = selected_id;
  selected_logprob[0] = logits[selected_id] / temperature - logsumexp;
  rng_word[0] = word;
}

#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
__global__ void dspark_draft_init_kernel(
    const std::uint32_t* input_ids, const __nv_bfloat16* embedding,
    std::uint32_t* draft_ids, __nv_bfloat16* output,
    std::uint32_t* error, std::uint64_t elements,
    std::uint32_t noise_id, std::uint32_t block_size,
    std::uint32_t vocab_size, std::uint32_t hidden_size) {
  const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
  if (index >= elements) return;
  const auto column = index % hidden_size;
  const auto stream_row = index / hidden_size;
  const auto draft_row = stream_row / 4U;
  const auto position = draft_row % block_size;
  const auto sequence = draft_row / block_size;
  const auto id = position == 0 ? input_ids[sequence] : noise_id;
  if (id >= vocab_size) {
    atomicOr(error, 16384U);
    output[index] = __float2bfloat16_rn(0.0F);
    return;
  }
  if (column == 0 && (stream_row % 4U) == 0) draft_ids[draft_row] = id;
  const auto value = embedding[static_cast<std::uint64_t>(id) * hidden_size +
                               column];
  if (!isfinite(__bfloat162float(value))) atomicOr(error, 32768U);
  output[index] = value;
}
#endif

Status before_launch(const char* operation) {
  return cuda_status(cudaPeekAtLastError(), operation);
}
}  // namespace pih::<anonymous>

Status launch_deepseek_embedding(DeepSeekEmbeddingLaunch launch) {
  auto status = validate_deepseek_embedding_launch(launch);
  if (!status.ok()) return status;
  status = before_launch("cudaPeekAtLastError before DeepSeek embedding");
  if (!status.ok()) return status;
  const auto elements = static_cast<std::uint64_t>(launch.token_count) * 4U *
                        launch.hidden_size;
  embedding_kernel<<<static_cast<std::uint32_t>((elements + 255U) / 256U), 256,
                     0, reinterpret_cast<cudaStream_t>(launch.stream)>>>(
      reinterpret_cast<const std::uint32_t*>(launch.token_ids_u32),
      reinterpret_cast<const __nv_bfloat16*>(launch.weight_bf16),
      reinterpret_cast<__nv_bfloat16*>(launch.output_hc_bf16),
      reinterpret_cast<std::uint32_t*>(launch.error_flag_u32), elements,
      launch.hidden_size, launch.vocab_size);
  return cuda_status(cudaPeekAtLastError(), "DeepSeek embedding launch");
}

Status launch_deepseek_hc_head(DeepSeekHcHeadLaunch launch) {
  auto status = validate_deepseek_hc_head_launch(launch);
  if (!status.ok()) return status;
  status = before_launch("cudaPeekAtLastError before DeepSeek HC head");
  if (!status.ok()) return status;
  hc_head_kernel<<<launch.token_count, 256, 0,
                   reinterpret_cast<cudaStream_t>(launch.stream)>>>(
      reinterpret_cast<const __nv_bfloat16*>(launch.input_hc_bf16),
      reinterpret_cast<const float*>(launch.fn_f32),
      reinterpret_cast<const float*>(launch.scale_f32),
      reinterpret_cast<const float*>(launch.base_f32),
      reinterpret_cast<__nv_bfloat16*>(launch.output_bf16),
      reinterpret_cast<std::uint32_t*>(launch.error_flag_u32),
      launch.hidden_size, launch.rms_epsilon, launch.hc_epsilon);
  return cuda_status(cudaPeekAtLastError(), "DeepSeek HC head launch");
}

Status launch_deepseek_lm_head(DeepSeekLmHeadLaunch launch) {
  auto status = validate_deepseek_lm_head_launch(launch);
  if (!status.ok()) return status;
  status = before_launch("cudaPeekAtLastError before DeepSeek LM head");
  if (!status.ok()) return status;
  lm_head_kernel<<<dim3(launch.vocab_size, launch.output_rows), 256, 0,
                     reinterpret_cast<cudaStream_t>(launch.stream)>>>(
      reinterpret_cast<const __nv_bfloat16*>(launch.input_bf16),
      reinterpret_cast<const __nv_bfloat16*>(launch.weight_bf16),
      reinterpret_cast<float*>(launch.logits_f32),
      reinterpret_cast<std::uint32_t*>(launch.error_flag_u32),
      launch.hidden_size, launch.vocab_size);
  return cuda_status(cudaPeekAtLastError(), "DeepSeek LM head launch");
}

#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
Status launch_deepseek_dspark_markov(DeepSeekDsparkMarkovLaunch launch) {
  auto status = validate_deepseek_dspark_markov_launch(launch);
  if (!status.ok()) return status;
  status = before_launch("cudaPeekAtLastError before DeepSeek DSpark Markov");
  if (!status.ok()) return status;
  dspark_markov_kernel<<<dim3(launch.vocab_size, launch.row_count), 256, 0,
      reinterpret_cast<cudaStream_t>(launch.stream)>>>(
      reinterpret_cast<const std::uint32_t*>(launch.token_ids_u32),
      reinterpret_cast<const __nv_bfloat16*>(launch.embedding_weight_bf16),
      reinterpret_cast<const __nv_bfloat16*>(launch.head_weight_bf16),
      reinterpret_cast<const float*>(launch.raw_logits_f32),
      reinterpret_cast<__nv_bfloat16*>(launch.markov_embeddings_bf16),
      reinterpret_cast<float*>(launch.biased_logits_f32),
      reinterpret_cast<std::uint32_t*>(launch.error_flag_u32),
      launch.vocab_size, launch.markov_rank);
  return cuda_status(cudaPeekAtLastError(), "DeepSeek DSpark Markov launch");
}

Status launch_deepseek_dspark_confidence(
    DeepSeekDsparkConfidenceLaunch launch) {
  auto status = validate_deepseek_dspark_confidence_launch(launch);
  if (!status.ok()) return status;
  status = before_launch(
      "cudaPeekAtLastError before DeepSeek DSpark confidence");
  if (!status.ok()) return status;
  dspark_confidence_kernel<<<launch.row_count, 256, 0,
      reinterpret_cast<cudaStream_t>(launch.stream)>>>(
      reinterpret_cast<const __nv_bfloat16*>(launch.hidden_bf16),
      reinterpret_cast<const __nv_bfloat16*>(launch.markov_embeddings_bf16),
      reinterpret_cast<const __nv_bfloat16*>(launch.projection_weight_bf16),
      reinterpret_cast<float*>(launch.confidence_f32),
      reinterpret_cast<std::uint32_t*>(launch.error_flag_u32),
      launch.hidden_size, launch.markov_rank);
  return cuda_status(cudaPeekAtLastError(),
                     "DeepSeek DSpark confidence launch");
}
#endif

Status launch_deepseek_argmax(DeepSeekArgmaxLaunch launch) {
  auto status = validate_deepseek_argmax_launch(launch);
  if (!status.ok()) return status;
  status = before_launch("cudaPeekAtLastError before DeepSeek argmax");
  if (!status.ok()) return status;
  argmax_kernel<<<1, 256, 0,
      reinterpret_cast<cudaStream_t>(launch.stream)>>>(
      reinterpret_cast<const float*>(launch.logits_f32),
      reinterpret_cast<std::uint32_t*>(launch.token_id_u32),
      reinterpret_cast<float*>(launch.selected_logprob_f32),
      reinterpret_cast<std::uint32_t*>(launch.top_logprobs_ids_u32),
      reinterpret_cast<float*>(launch.top_logprobs_f32),
      launch.top_logprobs_count,
      launch.suppressed_tokens,
      reinterpret_cast<std::uint32_t*>(launch.error_flag_u32),
      launch.vocab_size);
  return cuda_status(cudaPeekAtLastError(), "DeepSeek argmax launch");
}

Status launch_deepseek_stochastic_sample(
    DeepSeekStochasticSampleLaunch launch) {
  auto status = validate_deepseek_stochastic_sample_launch(launch);
  if (!status.ok()) return status;
  status = before_launch(
      "cudaPeekAtLastError before DeepSeek stochastic sample");
  if (!status.ok()) return status;
  stochastic_sample_compatibility_kernel<<<
      1, 1, 0, reinterpret_cast<cudaStream_t>(launch.stream)>>>(
      reinterpret_cast<const float*>(launch.logits_f32),
      reinterpret_cast<std::uint32_t*>(launch.token_id_u32),
      reinterpret_cast<float*>(launch.selected_logprob_f32),
      reinterpret_cast<std::uint32_t*>(launch.rng_word_u32),
      reinterpret_cast<float*>(launch.workspace_values_f32),
      reinterpret_cast<std::uint32_t*>(launch.workspace_ids_u32),
      reinterpret_cast<std::uint32_t*>(launch.error_flag_u32),
      launch.vocab_size, launch.temperature, launch.top_p, launch.top_k,
      launch.seed, launch.sample_ordinal,
      reinterpret_cast<std::uint32_t*>(launch.top_logprobs_ids_u32),
      reinterpret_cast<float*>(launch.top_logprobs_f32),
      launch.top_logprobs_count, launch.suppressed_tokens);
  return cuda_status(cudaPeekAtLastError(),
                     "DeepSeek stochastic sample launch");
}

#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
Status launch_deepseek_dspark_draft_init(
    DeepSeekDsparkDraftInitLaunch launch) {
  auto status = validate_deepseek_dspark_draft_init_launch(launch);
  if (!status.ok()) return status;
  status = before_launch(
      "cudaPeekAtLastError before DeepSeek DSpark draft init");
  if (!status.ok()) return status;
  const auto elements = static_cast<std::uint64_t>(launch.sequence_count) *
                        launch.block_size * launch.hc_multiplicity *
                        launch.hidden_size;
  dspark_draft_init_kernel<<<
      static_cast<std::uint32_t>((elements + 255U) / 256U), 256, 0,
      reinterpret_cast<cudaStream_t>(launch.stream)>>>(
      reinterpret_cast<const std::uint32_t*>(launch.input_token_ids_u32),
      reinterpret_cast<const __nv_bfloat16*>(launch.embedding_weight_bf16),
      reinterpret_cast<std::uint32_t*>(launch.draft_token_ids_u32),
      reinterpret_cast<__nv_bfloat16*>(launch.output_hc_bf16),
      reinterpret_cast<std::uint32_t*>(launch.error_flag_u32), elements,
      launch.noise_token_id, launch.block_size, launch.vocab_size,
      launch.hidden_size);
  return cuda_status(cudaPeekAtLastError(),
                     "DeepSeek DSpark draft init launch");
}
#endif

}  // namespace pih
