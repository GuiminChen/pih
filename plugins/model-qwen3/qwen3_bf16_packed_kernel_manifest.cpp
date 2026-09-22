#include "pih/model/qwen3_bf16_packed_kernel_manifest.h"

#include <vector>

#include "pih/core/sha256.h"

namespace pih {
namespace {

constexpr std::string_view kEmbeddingDescriptor =
    "pih.typed_kernel_params.v1|qwen.embedding_bf16.packed.v2|table:"
    "device_pointer_u64@0|token_ids:device_pointer_u64@8|output:device_"
    "pointer_u64@16|real_token_count:u64@24|vocabulary_size:u64@32|hidden_"
    "size:u64@40|total:48|layout:packed_token_u32_real_prefix_v1";
constexpr std::string_view kRopeAnglesDescriptor =
    "pih.typed_kernel_params.v1|qwen.rope_angles_f32.packed.v2|positions:"
    "device_pointer_u64@0|cosine:device_pointer_u64@8|sine:device_pointer_u64@"
    "16|real_token_count:u64@24|head_dim:u64@32|total:40|layout:packed_"
    "position_u64_real_prefix_v1";
constexpr std::string_view kKvAppendDescriptor =
    "pih.typed_kernel_params.v1|qwen.kv_append_bf16.packed.v2|key_input:"
    "device_pointer_u64@0|value_input:device_pointer_u64@8|kv_backing:device_"
    "pointer_u64@16|slot_states:device_pointer_u64@24|handles:device_pointer_"
    "u64@32|token_offsets:device_pointer_u64@40|request_index:device_pointer_"
    "u64@48|owner_sequence_indices:device_pointer_u64@56|error_flag:device_"
    "pointer_u64@64|layer:u32@72|token_count:u64@80|sequence_count:u32@88|"
    "slot_count:u32@92|total:96|layout:packed_owner_checked_slot16_v1";
constexpr std::string_view kPagedGqaDescriptor =
    "pih.typed_kernel_params.v1|qwen.paged_gqa_bf16.packed.v3|query:"
    "device_pointer_u64@0|output:device_pointer_u64@8|kv_backing:device_pointer_"
    "u64@16|slot_states:device_pointer_u64@24|handles:device_pointer_u64@32|"
    "visible_handle_offsets:device_pointer_u64@40|request_index:device_pointer_"
    "u64@48|query_start_offsets:device_pointer_u64@56|key_token_counts:device_"
    "pointer_u64@64|owner_sequence_indices:device_pointer_u64@72|error_flag:"
    "device_pointer_u64@80|layer:u32@88|query_count:u32@92|sequence_count:u32@"
    "96|scale:float32@100|slot_count:u32@104|total:108|reduction:packed_causal_"
    "online_softmax_head128_v3";
constexpr std::string_view kSampleHiddenDescriptor =
    "pih.typed_kernel_params.v1|qwen.sample_hidden_bf16.packed.v1|input:"
    "device_pointer_u64@0|sample_rows:device_pointer_u64@8|output:device_"
    "pointer_u64@16|error_flag:device_pointer_u64@24|sample_count:u32@32|"
    "packed_token_count:u32@36|hidden_size:u32@40|total:44|layout:packed_"
    "sample_rows_u32_v1";
constexpr std::string_view kGreedyArgmaxDescriptor =
    "pih.typed_kernel_params.v1|qwen.greedy_argmax_f32.packed.v2|logits:"
    "device_pointer_u64@0|sampled_token_ids:device_pointer_u64@8|error_flag:"
    "device_pointer_u64@16|sample_count:u32@24|vocabulary_size:u32@28|total:"
    "32|reduction:lowest_token_tie_per_sample_v2";
constexpr std::string_view kSamplerDescriptor =
    "pih.typed_kernel_params.v1|qwen.sample_f32.packed.v1|logits:device_"
    "pointer_u64@0|sampling_descriptors:device_pointer_u64@8|sample_sequence_"
    "indices:device_pointer_u64@16|workspace_ids:device_pointer_u64@24|sampled_"
    "token_ids:device_pointer_u64@32|selected_logprobs:device_pointer_u64@40|"
    "rng_words:device_pointer_u64@48|top_token_ids:device_pointer_u64@56|top_"
    "logprobs:device_pointer_u64@64|top_counts:device_pointer_u64@72|error_flag:"
    "device_pointer_u64@80|sample_count:u32@88|sequence_count:u32@92|vocabulary_"
    "size:u32@96|total:100|reduction:pih_sampler_v1";

Result<std::string> descriptor_digest(std::string_view descriptor) {
  auto digest = sha256(std::as_bytes(std::span(descriptor)));
  if (!digest.ok()) return digest.status();
  return digest->hex();
}

}  // namespace

Result<std::string_view> qwen_bf16_packed_kernel_symbol(
    QwenBf16PackedPrimitive primitive) {
  switch (primitive) {
    case QwenBf16PackedPrimitive::kEmbedding:
      return std::string_view("pih_qwen_embedding_bf16_packed_v2");
    case QwenBf16PackedPrimitive::kRopeAngles:
      return std::string_view("pih_qwen_rope_angles_f32_packed_v2");
    case QwenBf16PackedPrimitive::kKvAppend:
      return std::string_view("pih_qwen_kv_append_bf16_packed_v2");
    case QwenBf16PackedPrimitive::kPagedGqa:
      return std::string_view("pih_qwen_paged_gqa_bf16_packed_v3");
    case QwenBf16PackedPrimitive::kSampleHidden:
      return std::string_view("pih_qwen_sample_hidden_bf16_packed_v1");
    case QwenBf16PackedPrimitive::kGreedyArgmax:
      return std::string_view("pih_qwen_greedy_argmax_f32_packed_v2");
    case QwenBf16PackedPrimitive::kSampler:
      return std::string_view("pih_qwen_sample_f32_packed_v1");
  }
  return Status::InvalidArgument("unknown packed Qwen primitive");
}

Result<std::string_view> qwen_bf16_packed_parameter_abi_descriptor(
    QwenBf16PackedPrimitive primitive) {
  switch (primitive) {
    case QwenBf16PackedPrimitive::kEmbedding: return kEmbeddingDescriptor;
    case QwenBf16PackedPrimitive::kRopeAngles: return kRopeAnglesDescriptor;
    case QwenBf16PackedPrimitive::kKvAppend: return kKvAppendDescriptor;
    case QwenBf16PackedPrimitive::kPagedGqa: return kPagedGqaDescriptor;
    case QwenBf16PackedPrimitive::kSampleHidden: return kSampleHiddenDescriptor;
    case QwenBf16PackedPrimitive::kGreedyArgmax: return kGreedyArgmaxDescriptor;
    case QwenBf16PackedPrimitive::kSampler: return kSamplerDescriptor;
  }
  return Status::InvalidArgument("unknown packed Qwen primitive");
}

Result<KernelSignatureManifest> qwen_bf16_packed_kernel_manifest(
    QwenBf16PackedPrimitive primitive, std::string selected_cubin_sha256) {
  auto descriptor = qwen_bf16_packed_parameter_abi_descriptor(primitive);
  if (!descriptor.ok()) return descriptor.status();
  auto digest = descriptor_digest(*descriptor);
  if (!digest.ok()) return digest.status();
  std::string logical_id;
  std::vector<KernelParameterSpec> parameters;
  std::uint32_t total = 0;
  using Wire = KernelWireType;
  if (primitive == QwenBf16PackedPrimitive::kEmbedding) {
    logical_id = "qwen.embedding_bf16.packed.v2";
    parameters = {
        {"table", Wire::kDevicePointerU64, 0, "qwen_embedding_bf16_read_v1"},
        {"token_ids", Wire::kDevicePointerU64, 8, "packed_token_u32_read_v1"},
        {"output", Wire::kDevicePointerU64, 16, "packed_hidden_bf16_write_v1"},
        {"real_token_count", Wire::kU64, 24, "packed_real_token_count_v1"},
        {"vocabulary_size", Wire::kU64, 32, "qwen_vocabulary_exact_v1"},
        {"hidden_size", Wire::kU64, 40, "qwen_hidden_size_exact_v1"}};
    total = 48;
  } else if (primitive == QwenBf16PackedPrimitive::kRopeAngles) {
    logical_id = "qwen.rope_angles_f32.packed.v2";
    parameters = {
        {"positions", Wire::kDevicePointerU64, 0, "packed_position_u64_read_v1"},
        {"cosine", Wire::kDevicePointerU64, 8, "qwen_rope_cosine_f32_write_v1"},
        {"sine", Wire::kDevicePointerU64, 16, "qwen_rope_sine_f32_write_v1"},
        {"real_token_count", Wire::kU64, 24, "packed_real_token_count_v1"},
        {"head_dim", Wire::kU64, 32, "qwen_head_dim_128_v1"}};
    total = 40;
  } else if (primitive == QwenBf16PackedPrimitive::kKvAppend) {
    logical_id = "qwen.kv_append_bf16.packed.v2";
    parameters = {
        {"key_input", Wire::kDevicePointerU64, 0, "packed_key_bf16_read_v1"},
        {"value_input", Wire::kDevicePointerU64, 8, "packed_value_bf16_read_v1"},
        {"kv_backing", Wire::kDevicePointerU64, 16, "qwen_kv_bf16_write_v1"},
        {"slot_states", Wire::kDevicePointerU64, 24, "qwen_kv_state16_read_v1"},
        {"handles", Wire::kDevicePointerU64, 32, "packed_kv_handle8_read_v1"},
        {"token_offsets", Wire::kDevicePointerU64, 40, "packed_kv_offset_u16_read_v1"},
        {"request_index", Wire::kDevicePointerU64, 48, "packed_request_index_u32_read_v1"},
        {"owner_sequence_indices", Wire::kDevicePointerU64, 56, "packed_owner_index_u32_read_v1"},
        {"error_flag", Wire::kDevicePointerU64, 64, "qwen_error_u32_atomic_v1"},
        {"layer", Wire::kU32, 72, "qwen_layer_0_27_v1"},
        {"token_count", Wire::kU64, 80, "packed_real_token_count_v1"},
        {"sequence_count", Wire::kU32, 88, "packed_sequence_count_v1"},
        {"slot_count", Wire::kU32, 92, "qwen_kv_slot_count_v1"}};
    total = 96;
  } else if (primitive == QwenBf16PackedPrimitive::kPagedGqa) {
    logical_id = "qwen.paged_gqa_bf16.packed.v3";
    parameters = {
        {"query", Wire::kDevicePointerU64, 0, "packed_query_bf16_read_v1"},
        {"output", Wire::kDevicePointerU64, 8, "packed_attention_bf16_write_v1"},
        {"kv_backing", Wire::kDevicePointerU64, 16, "qwen_kv_bf16_read_v1"},
        {"slot_states", Wire::kDevicePointerU64, 24, "qwen_kv_state16_read_v1"},
        {"handles", Wire::kDevicePointerU64, 32, "packed_visible_handle8_read_v1"},
        {"visible_handle_offsets", Wire::kDevicePointerU64, 40, "packed_visible_prefix_u32_read_v1"},
        {"request_index", Wire::kDevicePointerU64, 48, "packed_request_index_u32_read_v1"},
        {"query_start_offsets", Wire::kDevicePointerU64, 56, "packed_query_prefix_u32_read_v1"},
        {"key_token_counts", Wire::kDevicePointerU64, 64, "packed_key_count_u32_read_v1"},
        {"owner_sequence_indices", Wire::kDevicePointerU64, 72, "packed_owner_index_u32_read_v1"},
        {"error_flag", Wire::kDevicePointerU64, 80, "qwen_error_u32_atomic_v1"},
        {"layer", Wire::kU32, 88, "qwen_layer_0_27_v1"},
        {"query_count", Wire::kU32, 92, "packed_real_token_count_v1"},
        {"sequence_count", Wire::kU32, 96, "packed_sequence_count_v1"},
        {"scale", Wire::kFloat32, 100, "qwen_attention_scale_v1"},
        {"slot_count", Wire::kU32, 104, "qwen_kv_slot_count_v1"}};
    total = 108;
  } else if (primitive == QwenBf16PackedPrimitive::kSampleHidden) {
    logical_id = "qwen.sample_hidden_bf16.packed.v1";
    parameters = {
        {"input", Wire::kDevicePointerU64, 0, "packed_hidden_bf16_read_v1"},
        {"sample_rows", Wire::kDevicePointerU64, 8, "packed_sample_rows_u32_read_v1"},
        {"output", Wire::kDevicePointerU64, 16, "sample_hidden_bf16_write_v1"},
        {"error_flag", Wire::kDevicePointerU64, 24, "qwen_error_u32_atomic_v1"},
        {"sample_count", Wire::kU32, 32, "packed_sample_count_v1"},
        {"packed_token_count", Wire::kU32, 36, "packed_execution_bucket_v1"},
        {"hidden_size", Wire::kU32, 40, "qwen_hidden_size_exact_v1"}};
    total = 44;
  } else if (primitive == QwenBf16PackedPrimitive::kGreedyArgmax) {
    logical_id = "qwen.greedy_argmax_f32.packed.v2";
    parameters = {
        {"logits", Wire::kDevicePointerU64, 0, "packed_logits_f32_read_v1"},
        {"sampled_token_ids", Wire::kDevicePointerU64, 8, "packed_sampled_u32_write_v1"},
        {"error_flag", Wire::kDevicePointerU64, 16, "qwen_error_u32_atomic_v1"},
        {"sample_count", Wire::kU32, 24, "packed_sample_count_v1"},
        {"vocabulary_size", Wire::kU32, 28, "qwen_vocabulary_exact_v1"}};
    total = 32;
  } else if (primitive == QwenBf16PackedPrimitive::kSampler) {
    logical_id = "qwen.sample_f32.packed.v1";
    parameters = {
        {"logits", Wire::kDevicePointerU64, 0, "packed_logits_f32_mutate_v1"},
        {"sampling_descriptors", Wire::kDevicePointerU64, 8, "packed_sampling112_read_v2"},
        {"sample_sequence_indices", Wire::kDevicePointerU64, 16, "packed_sample_sequence_u32_read_v1"},
        {"workspace_ids", Wire::kDevicePointerU64, 24, "packed_sampler_ids_u32_write_v1"},
        {"sampled_token_ids", Wire::kDevicePointerU64, 32, "packed_sampled_u32_write_v1"},
        {"selected_logprobs", Wire::kDevicePointerU64, 40, "packed_selected_logprob_f32_write_v1"},
        {"rng_words", Wire::kDevicePointerU64, 48, "packed_rng_u32_write_v1"},
        {"top_token_ids", Wire::kDevicePointerU64, 56, "packed_top_ids_u32_write_v1"},
        {"top_logprobs", Wire::kDevicePointerU64, 64, "packed_top_logprobs_f32_write_v1"},
        {"top_counts", Wire::kDevicePointerU64, 72, "packed_top_counts_u32_write_v1"},
        {"error_flag", Wire::kDevicePointerU64, 80, "qwen_error_u32_atomic_v1"},
        {"sample_count", Wire::kU32, 88, "packed_sample_count_v1"},
        {"sequence_count", Wire::kU32, 92, "packed_sequence_count_v1"},
        {"vocabulary_size", Wire::kU32, 96, "qwen_vocabulary_exact_v1"}};
    total = 100;
  } else {
    return Status::InvalidArgument("unknown packed Qwen primitive");
  }
  return KernelSignatureManifest::Create(
      std::move(logical_id), std::move(selected_cubin_sha256),
      std::move(*digest), parameters, total);
}

}  // namespace pih
