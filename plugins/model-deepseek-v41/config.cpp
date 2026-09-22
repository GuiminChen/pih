#include "config.h"
#include "pih/core/bounded_json.h"
#include <algorithm>
#include <string>

namespace pih::deepseek_v41 {
namespace {
// Frozen semantic contract from the vendored revision's hf_config.json. This is
// metadata, not a Python import or an execution fallback. Reject unknown fields
// rather than silently ignoring future execution-affecting configuration.
constexpr std::string_view kConfiguration = R"json({
"architectures":["DeepseekV41ForCausalLM"],"model_type":"deepseek_v41","dtype":"bfloat16",
"transformers_version":"5.6.0","bos_token_id":0,"eos_token_id":1,"pad_token_id":2,"image_token_id":129264,
"quantization_config":{"quant_method":"fp8","activation_scheme":"dynamic","weight_block_size":[32,32],"scale_fmt":"ue8m0","expert_dtype":"fp4"},
"text_config":{
"model_type":"deepseek_v41_text","vocab_size":129280,"hidden_size":5120,"moe_intermediate_size":2304,
"num_hidden_layers":40,"num_attention_heads":64,"num_key_value_heads":1,"head_dim":512,
"qk_rope_head_dim":64,"q_lora_rank":1280,"o_lora_rank":1024,"o_groups":8,"hidden_act":"silu",
"swiglu_limit":10.0,"rms_norm_eps":1e-20,"attention_bias":false,"attention_dropout":0.0,
"initializer_range":0.02,"use_cache":true,"tie_word_embeddings":false,"max_position_embeddings":1048576,
"rope_theta":10000,"rope_scaling":{"rope_type":"yarn","factor":16,"beta_fast":32,"beta_slow":1,"original_max_position_embeddings":65536},
"n_routed_experts":384,"n_shared_experts":1,"num_experts_per_tok":6,"scoring_func":"sqrtsoftplus",
"topk_method":"noaux_tc","norm_topk_prob":true,"routed_scaling_factor":1.5,"sliding_window":128,
"compress_ratios":[0,0,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
"compress_rope_theta":160000,"kv_source_layer_ids":[2,8,14,20],"index_source_layer_ids":[2,8,14,20,24,28,32,36],
"index_n_heads":32,"index_head_dim":128,"index_topk":512,"candidate_source_layer_id":20,
"candidate_topk_blocks":2048,"candidate_block_size":8,"hc_mult":4,"hc_sinkhorn_iters":20,"hc_eps":1e-6,
"engram_layer_ids":[1,14],"engram_num_embeddings":[384006168,384016682],"engram_max_ngram_size":4,
"engram_vocab_size":16000000,"engram_n_heads":8,"engram_head_dim":256,"engram_pad_token_id":2,
"engram_compressed_vocab_size":99092,"num_nextn_predict_layers":3,"dspark_block_size":5,
"dspark_noise_token_id":128799,"dspark_target_layer_ids":[37,38,39],"dspark_markov_rank":256,
"dspark_n_routed_experts":128,"dspark_num_experts_per_tok":3},
"vision_config":{"model_type":"deepseek_v41_vision","num_hidden_layers":32,"hidden_size":1024,
"num_attention_heads":16,"intermediate_size":2816,"patch_size":14,"rope_theta":10000,
"downsample_ratio":3,"max_image_tokens":1024,"min_pixels":295936,"max_wh_ratio":null}
})json";
Status Compare(const JsonValue& actual, const JsonValue& expected, const std::string& path) {
  const auto mismatch = [&] { return Status::InvalidArgument("V4.1 Flash configuration mismatch at " + path); };
  if (expected.is_object()) {
    if (!actual.is_object() || actual.object().size() != expected.object().size()) return mismatch();
    for (const auto& [name, value] : expected.object()) {
      const auto* observed = actual.at(name);
      if (!observed) return mismatch();
      auto status = Compare(*observed, value, path + "." + name);
      if (!status.ok()) return status;
    }
  } else if (expected.is_array()) {
    if (!actual.is_array() || actual.array().size() != expected.array().size()) return mismatch();
    for (std::size_t i = 0; i < expected.array().size(); ++i) {
      auto status = Compare(actual.array()[i], expected.array()[i], path + "[" + std::to_string(i) + "]");
      if (!status.ok()) return status;
    }
  } else if (expected.is_integer()) {
    if (!actual.is_integer() || actual.integer() != expected.integer()) return mismatch();
  } else if (expected.is_number()) {
    if (!actual.is_number() || actual.number() != expected.number()) return mismatch();
  } else if (expected.is_boolean()) {
    if (!actual.is_boolean() || actual.boolean() != expected.boolean()) return mismatch();
  } else if (expected.is_string()) {
    if (!actual.is_string() || actual.string() != expected.string()) return mismatch();
  } else if (!actual.is_null()) return mismatch();
  return Status::Ok();
}
}  // namespace
Result<FlashConfig> FlashConfig::Parse(std::string_view json) {
  JsonLimits limits;
  limits.max_input_bytes = kMaximumBytes;
  limits.max_depth = 8;
  limits.max_nodes = 1024;
  limits.max_string_bytes = 4096;
  auto actual = JsonValue::Parse(json, limits);
  if (!actual.ok()) return actual.status();
  auto expected = JsonValue::Parse(kConfiguration, limits);
  if (!expected.ok()) return Status::Internal("frozen V4.1 configuration contract malformed");
  auto status = Compare(*actual, *expected, "config");
  if (!status.ok()) return status;
  FlashConfig result;
  auto digest = sha256(std::as_bytes(std::span(json)));
  if (!digest.ok()) return digest.status();
  result.config_sha256_ = *digest;
  constexpr std::array<std::uint32_t, 4> kv_sources{2, 8, 14, 20};
  constexpr std::array<std::uint32_t, 8> index_sources{2, 8, 14, 20, 24, 28, 32, 36};
  std::uint32_t kv = UINT32_MAX, index = UINT32_MAX;
  for (std::uint32_t layer = 0; layer < result.layers_.size(); ++layer) {
    auto& plan = result.layers_[layer];
    plan.layer = layer;
    plan.compression_ratio = layer < 2 || layer >= 40 ? 0 : layer < 20 ? 2 : 1;
    plan.owns_kv = std::ranges::find(kv_sources, layer) != kv_sources.end();
    plan.owns_index = std::ranges::find(index_sources, layer) != index_sources.end();
    if (plan.owns_kv) kv = layer;
    if (plan.owns_index) index = layer;
    if (plan.compression_ratio) { plan.kv_source = kv; plan.index_source = index; }
    plan.produces_candidates = layer == 20;
    plan.index_uses_candidates = plan.owns_index && layer > 20;
    plan.has_engram = layer == 1 || layer == 14;
    plan.routed_experts = layer < 40 ? 384 : 128;
    plan.activated_experts = layer < 40 ? 6 : 3;
  }
  return result;
}
}  // namespace pih::deepseek_v41
