#include "pih/model/qwen3_bf16_kernel_manifest.h"

#include <array>
#include <vector>

namespace pih {
namespace {

constexpr std::string_view kEmbeddingDescriptor =
    "pih.typed_kernel_params.v1|qwen.embedding.bf16.v1|table:device_"
    "pointer_u64@0|token_ids:device_pointer_u64@8|output:device_pointer_u64@16|"
    "token_count:u64@24|vocabulary_size:u64@32|hidden_size:u64@40|total:48";
constexpr std::string_view kResidualDescriptor =
    "pih.typed_kernel_params.v1|qwen.residual_add.bf16.v1|lhs:device_"
    "pointer_u64@0|rhs:device_pointer_u64@8|output:device_pointer_u64@16|"
    "elements:u64@24|total:32";
constexpr std::string_view kSiluDescriptor =
    "pih.typed_kernel_params.v1|qwen.silu_mul.bf16.v1|gate:device_"
    "pointer_u64@0|up:device_pointer_u64@8|output:device_pointer_u64@16|"
    "elements:u64@24|total:32";
constexpr std::string_view kRmsNormDescriptor =
    "pih.typed_kernel_params.v1|qwen.rms_norm.bf16.v1|input:device_"
    "pointer_u64@0|weight:device_pointer_u64@8|output:device_pointer_u64@16|"
    "rows:u64@24|hidden_size:u64@32|epsilon:float32@40|total:44|reduction:"
    "qwen_hidden1024_block256_tree_v1";
constexpr std::string_view kRopeDescriptor =
    "pih.typed_kernel_params.v1|qwen.rope.bf16.v2|input:device_pointer_"
    "u64@0|cosine:device_pointer_u64@8|sine:device_pointer_u64@16|output:"
    "device_pointer_u64@24|vectors:u64@32|heads:u64@40|head_dim:u64@48|"
    "total:56|layout:token_broadcast_rotate_half_v1";
constexpr std::string_view kKvAppendDescriptor =
    "pih.typed_kernel_params.v1|qwen.kv_append.bf16.v1|key_input:device_"
    "pointer_u64@0|value_input:device_pointer_u64@8|kv_backing:device_pointer_"
    "u64@16|slot_states:device_pointer_u64@24|handles:device_pointer_u64@32|"
    "token_offsets:device_pointer_u64@40|error_flag:device_pointer_u64@48|"
    "owner_sequence_index:u32@56|layer:u32@60|token_count:u64@64|slot_count:"
    "u32@72|total:76|layout:qwen_kv_slot16_layer_plane_token_head_column_v1";
constexpr std::string_view kPagedGqaDescriptor =
    "pih.typed_kernel_params.v1|qwen.paged_gqa.bf16.v2|query:device_"
    "pointer_u64@0|output:device_pointer_u64@8|kv_backing:device_pointer_u64@16|"
    "slot_states:device_pointer_u64@24|handles:device_pointer_u64@32|error_flag:"
    "device_pointer_u64@40|owner_sequence_index:u32@48|layer:u32@52|query_"
    "start_position:u64@56|query_count:u32@64|handle_count:u32@68|key_token_"
    "count:u32@72|scale:float32@76|slot_count:u32@80|total:84|reduction:"
    "causal_online_softmax_head128_v2";
constexpr std::string_view kRopeAnglesDescriptor =
    "pih.typed_kernel_params.v1|qwen.rope_angles.f32.v1|positions:device_"
    "pointer_u64@0|cosine:device_pointer_u64@8|sine:device_pointer_u64@16|"
    "token_count:u64@24|head_dim:u64@32|total:40|layout:"
    "token_pair_fp32_theta1000000_v1";
constexpr std::string_view kGreedyArgmaxDescriptor =
    "pih.typed_kernel_params.v1|qwen.greedy_argmax.f32.v1|logits:device_"
    "pointer_u64@0|sampled_token:device_pointer_u64@8|error_flag:device_pointer_"
    "u64@16|vocabulary_size:u64@24|total:32|reduction:"
    "lowest_token_tie_block256_v1";
constexpr std::string_view kTeacherForcedMetricDescriptor =
    "pih.typed_kernel_params.v1|qwen.teacher_forced_metric.f32.v1|"
    "logits:device_pointer_u64@0|target_tokens:device_pointer_u64@8|argmax_"
    "tokens:device_pointer_u64@16|target_nll:device_pointer_u64@24|nonfinite_"
    "rows:device_pointer_u64@32|error_flag:device_pointer_u64@40|rows:u32@48|"
    "vocabulary_size:u32@52|total:56|reduction:lowest_token_tie_block256_fp64_"
    "nll_v1";

}  // namespace

Result<std::string_view> qwen_bf16_kernel_symbol(
    QwenBf16Primitive primitive) {
  switch (primitive) {
    case QwenBf16Primitive::kEmbedding:
      return std::string_view("pih_qwen_embedding_bf16_v1");
    case QwenBf16Primitive::kResidualAdd:
      return std::string_view("pih_qwen_residual_add_bf16_v1");
    case QwenBf16Primitive::kSiluMul:
      return std::string_view("pih_qwen_silu_mul_bf16_v1");
    case QwenBf16Primitive::kRmsNorm:
      return std::string_view("pih_qwen_rms_norm_bf16_v1");
    case QwenBf16Primitive::kRope:
      return std::string_view("pih_qwen_rope_bf16_v2");
    case QwenBf16Primitive::kKvAppend:
      return std::string_view("pih_qwen_kv_append_bf16_v1");
    case QwenBf16Primitive::kPagedGqa:
      return std::string_view("pih_qwen_paged_gqa_bf16_v2");
    case QwenBf16Primitive::kRopeAngles:
      return std::string_view("pih_qwen_rope_angles_f32_v1");
    case QwenBf16Primitive::kGreedyArgmax:
      return std::string_view("pih_qwen_greedy_argmax_f32_v1");
    case QwenBf16Primitive::kTeacherForcedMetric:
      return std::string_view(
          "pih_qwen_teacher_forced_metric_f32_v1");
  }
  return Status::InvalidArgument("unknown Qwen BF16 primitive");
}

Result<std::string_view> qwen_bf16_parameter_abi_descriptor(
    QwenBf16Primitive primitive) {
  switch (primitive) {
    case QwenBf16Primitive::kEmbedding:
      return kEmbeddingDescriptor;
    case QwenBf16Primitive::kResidualAdd:
      return kResidualDescriptor;
    case QwenBf16Primitive::kSiluMul:
      return kSiluDescriptor;
    case QwenBf16Primitive::kRmsNorm:
      return kRmsNormDescriptor;
    case QwenBf16Primitive::kRope:
      return kRopeDescriptor;
    case QwenBf16Primitive::kKvAppend:
      return kKvAppendDescriptor;
    case QwenBf16Primitive::kPagedGqa:
      return kPagedGqaDescriptor;
    case QwenBf16Primitive::kRopeAngles:
      return kRopeAnglesDescriptor;
    case QwenBf16Primitive::kGreedyArgmax:
      return kGreedyArgmaxDescriptor;
    case QwenBf16Primitive::kTeacherForcedMetric:
      return kTeacherForcedMetricDescriptor;
  }
  return Status::InvalidArgument("unknown Qwen BF16 primitive");
}

Result<KernelSignatureManifest> qwen_bf16_kernel_manifest(
    QwenBf16Primitive primitive, std::string selected_cubin_sha256) {
  std::string logical_id;
  std::string parameter_digest;
  std::vector<KernelParameterSpec> parameters;
  std::uint32_t total_bytes = 0;
  switch (primitive) {
    case QwenBf16Primitive::kEmbedding:
      logical_id = "qwen.embedding.bf16.v1";
      parameter_digest =
          "bad732f377e56e1db7b11a20290c403bed05b54c8cbabf42ca5db3c4d826594d";
      parameters = {
          {"table", KernelWireType::kDevicePointerU64, 0,
           "qwen_embedding_table_bf16_read_v1"},
          {"token_ids", KernelWireType::kDevicePointerU64, 8,
           "qwen_token_ids_i64_read_v1"},
          {"output", KernelWireType::kDevicePointerU64, 16,
           "qwen_embedding_output_bf16_write_v1"},
          {"token_count", KernelWireType::kU64, 24,
           "qwen_token_count_nonzero_v1"},
          {"vocabulary_size", KernelWireType::kU64, 32,
           "qwen_vocabulary_size_exact_v1"},
          {"hidden_size", KernelWireType::kU64, 40,
           "qwen_hidden_size_exact_v1"}};
      total_bytes = 48;
      break;
    case QwenBf16Primitive::kResidualAdd:
      logical_id = "qwen.residual_add.bf16.v1";
      parameter_digest =
          "026796cc032f6831719bd9df404dcdda22f1e73526f61de290a48fd6d83ebc3a";
      parameters = {
          {"lhs", KernelWireType::kDevicePointerU64, 0,
           "qwen_residual_lhs_bf16_read_v1"},
          {"rhs", KernelWireType::kDevicePointerU64, 8,
           "qwen_residual_rhs_bf16_read_v1"},
          {"output", KernelWireType::kDevicePointerU64, 16,
           "qwen_residual_output_bf16_write_v1"},
          {"elements", KernelWireType::kU64, 24,
           "qwen_element_count_nonzero_v1"}};
      total_bytes = 32;
      break;
    case QwenBf16Primitive::kSiluMul:
      logical_id = "qwen.silu_mul.bf16.v1";
      parameter_digest =
          "33bcbde0769cf73fb0d5beed95046814dbe6a652323b2d1873448082733998e4";
      parameters = {
          {"gate", KernelWireType::kDevicePointerU64, 0,
           "qwen_silu_gate_bf16_read_v1"},
          {"up", KernelWireType::kDevicePointerU64, 8,
           "qwen_silu_up_bf16_read_v1"},
          {"output", KernelWireType::kDevicePointerU64, 16,
           "qwen_silu_output_bf16_write_v1"},
          {"elements", KernelWireType::kU64, 24,
           "qwen_element_count_nonzero_v1"}};
      total_bytes = 32;
      break;
    case QwenBf16Primitive::kRmsNorm:
      logical_id = "qwen.rms_norm.bf16.v1";
      parameter_digest =
          "1b5ec9801a522d8c5cfbf4d47de85df5b89554deb754023ff5077d556f190457";
      parameters = {
          {"input", KernelWireType::kDevicePointerU64, 0,
           "qwen_rms_input_bf16_read_v1"},
          {"weight", KernelWireType::kDevicePointerU64, 8,
           "qwen_rms_weight_bf16_read_v1"},
          {"output", KernelWireType::kDevicePointerU64, 16,
           "qwen_rms_output_bf16_write_v1"},
          {"rows", KernelWireType::kU64, 24, "qwen_row_count_nonzero_v1"},
          {"hidden_size", KernelWireType::kU64, 32,
           "qwen_hidden_size_1024_v1"},
          {"epsilon", KernelWireType::kFloat32, 40,
           "qwen_rms_epsilon_positive_v1"}};
      total_bytes = 44;
      break;
    case QwenBf16Primitive::kRope:
      logical_id = "qwen.rope.bf16.v2";
      parameter_digest =
          "6219b4f6682fa0e303fe4db823dd5a47a497928c7a915156214cb329c0a3f7be";
      parameters = {
          {"input", KernelWireType::kDevicePointerU64, 0,
           "qwen_rope_input_bf16_read_v1"},
          {"cosine", KernelWireType::kDevicePointerU64, 8,
           "qwen_rope_cosine_f32_read_v1"},
          {"sine", KernelWireType::kDevicePointerU64, 16,
           "qwen_rope_sine_f32_read_v1"},
          {"output", KernelWireType::kDevicePointerU64, 24,
           "qwen_rope_output_bf16_write_v1"},
          {"vectors", KernelWireType::kU64, 32,
           "qwen_rope_vector_count_nonzero_v1"},
          {"heads", KernelWireType::kU64, 40,
           "qwen_rope_heads_8_or_16_v1"},
          {"head_dim", KernelWireType::kU64, 48,
           "qwen_head_dim_128_v1"}};
      total_bytes = 56;
      break;
    case QwenBf16Primitive::kKvAppend:
      logical_id = "qwen.kv_append.bf16.v1";
      parameter_digest =
          "e2a575834f28edeeb3b2f197f5508e717df0337195868068017110710686c36a";
      parameters = {
          {"key_input", KernelWireType::kDevicePointerU64, 0,
           "qwen_kv_append_key_bf16_read_v1"},
          {"value_input", KernelWireType::kDevicePointerU64, 8,
           "qwen_kv_append_value_bf16_read_v1"},
          {"kv_backing", KernelWireType::kDevicePointerU64, 16,
           "qwen_kv_backing_bf16_write_v1"},
          {"slot_states", KernelWireType::kDevicePointerU64, 24,
           "qwen_kv_slot_state16_read_v1"},
          {"handles", KernelWireType::kDevicePointerU64, 32,
           "qwen_kv_handle8_read_v1"},
          {"token_offsets", KernelWireType::kDevicePointerU64, 40,
           "qwen_kv_token_offset_u16_read_v1"},
          {"error_flag", KernelWireType::kDevicePointerU64, 48,
           "qwen_device_error_u32_atomic_v1"},
          {"owner_sequence_index", KernelWireType::kU32, 56,
           "qwen_sequence_owner_exact_v1"},
          {"layer", KernelWireType::kU32, 60, "qwen_layer_0_27_v1"},
          {"token_count", KernelWireType::kU64, 64,
           "qwen_token_count_nonzero_v1"},
          {"slot_count", KernelWireType::kU32, 72,
           "qwen_kv_slot_count_bounded_v1"}};
      total_bytes = 76;
      break;
    case QwenBf16Primitive::kPagedGqa:
      logical_id = "qwen.paged_gqa.bf16.v2";
      parameter_digest =
          "ce1dad2aa7af1f494ecf8b2839f24f8197d560dd5819a4980f11e85b04c0aa9d";
      parameters = {
          {"query", KernelWireType::kDevicePointerU64, 0,
           "qwen_gqa_query_bf16_16x128_read_v1"},
          {"output", KernelWireType::kDevicePointerU64, 8,
           "qwen_gqa_output_bf16_16x128_write_v1"},
          {"kv_backing", KernelWireType::kDevicePointerU64, 16,
           "qwen_kv_backing_bf16_read_v1"},
          {"slot_states", KernelWireType::kDevicePointerU64, 24,
           "qwen_kv_slot_state16_read_v1"},
          {"handles", KernelWireType::kDevicePointerU64, 32,
           "qwen_kv_handle8_read_v1"},
          {"error_flag", KernelWireType::kDevicePointerU64, 40,
           "qwen_device_error_u32_atomic_v1"},
          {"owner_sequence_index", KernelWireType::kU32, 48,
           "qwen_sequence_owner_exact_v1"},
          {"layer", KernelWireType::kU32, 52, "qwen_layer_0_27_v1"},
          {"query_start_position", KernelWireType::kU64, 56,
           "qwen_causal_query_start_position_v1"},
          {"query_count", KernelWireType::kU32, 64,
           "qwen_query_count_1_4096_v1"},
          {"handle_count", KernelWireType::kU32, 68,
           "qwen_kv_handle_count_bounded_v1"},
          {"key_token_count", KernelWireType::kU32, 72,
           "qwen_causal_key_token_count_v1"},
          {"scale", KernelWireType::kFloat32, 76,
           "qwen_attention_scale_positive_v1"},
          {"slot_count", KernelWireType::kU32, 80,
           "qwen_kv_slot_count_bounded_v1"}};
      total_bytes = 84;
      break;
    case QwenBf16Primitive::kRopeAngles:
      logical_id = "qwen.rope_angles.f32.v1";
      parameter_digest =
          "1d03e363fa823d7d5243b3ef434193a4b5a79b08f5751091e4097c409e2b79ac";
      parameters = {
          {"positions", KernelWireType::kDevicePointerU64, 0,
           "qwen_positions_i64_read_v1"},
          {"cosine", KernelWireType::kDevicePointerU64, 8,
           "qwen_rope_cosine_f32_write_v1"},
          {"sine", KernelWireType::kDevicePointerU64, 16,
           "qwen_rope_sine_f32_write_v1"},
          {"token_count", KernelWireType::kU64, 24,
           "qwen_token_count_nonzero_v1"},
          {"head_dim", KernelWireType::kU64, 32,
           "qwen_head_dim_128_v1"}};
      total_bytes = 40;
      break;
    case QwenBf16Primitive::kGreedyArgmax:
      logical_id = "qwen.greedy_argmax.f32.v1";
      parameter_digest =
          "c2c46fbae36f1211c9a74496d84b04277102ebeaa5eaadab24e81ff56ddbae56";
      parameters = {
          {"logits", KernelWireType::kDevicePointerU64, 0,
           "qwen_logits_f32_read_v1"},
          {"sampled_token", KernelWireType::kDevicePointerU64, 8,
           "qwen_sampled_token_i64_write_v1"},
          {"error_flag", KernelWireType::kDevicePointerU64, 16,
           "qwen_device_error_u32_atomic_v1"},
          {"vocabulary_size", KernelWireType::kU64, 24,
           "qwen_vocabulary_size_exact_v1"}};
      total_bytes = 32;
      break;
    case QwenBf16Primitive::kTeacherForcedMetric:
      logical_id = "qwen.teacher_forced_metric.f32.v1";
      parameter_digest =
          "872051209c59c60f287d6c8265a459fef03f6f4b296622a307d11cd892f15d91";
      parameters = {
          {"logits", KernelWireType::kDevicePointerU64, 0,
           "qwen_metric_logits_f32_read_v1"},
          {"target_tokens", KernelWireType::kDevicePointerU64, 8,
           "qwen_metric_targets_u32_read_v1"},
          {"argmax_tokens", KernelWireType::kDevicePointerU64, 16,
           "qwen_metric_argmax_u32_write_v1"},
          {"target_nll", KernelWireType::kDevicePointerU64, 24,
           "qwen_metric_target_nll_f64_write_v1"},
          {"nonfinite_rows", KernelWireType::kDevicePointerU64, 32,
           "qwen_metric_nonfinite_u32_write_v1"},
          {"error_flag", KernelWireType::kDevicePointerU64, 40,
           "qwen_device_error_u32_atomic_v1"},
          {"rows", KernelWireType::kU32, 48,
           "qwen_metric_rows_1_4096_v1"},
          {"vocabulary_size", KernelWireType::kU32, 52,
           "qwen_vocabulary_size_exact_v1"}};
      total_bytes = 56;
      break;
    default:
      return Status::InvalidArgument("unknown Qwen BF16 primitive");
  }
  return KernelSignatureManifest::Create(
      std::move(logical_id), std::move(selected_cubin_sha256),
      std::move(parameter_digest), parameters, total_bytes);
}

}  // namespace pih
