#include "pih/model/deepseek_v4_config.h"

#include <array>
#include <fstream>
#include <stdexcept>
#include <string>

#include "pih/core/bounded_json.h"

namespace pih {
namespace {

Status exact_integer(const JsonValue& root, std::string_view name,
                     std::int64_t expected) {
  const auto* value = root.at(name);
  if (value == nullptr || !value->is_integer() ||
      value->integer() != expected) {
    return Status::InvalidArgument(std::string(name) +
                                   " does not match DeepSeek V4 Flash 0731");
  }
  return Status::Ok();
}

Status exact_number(const JsonValue& root, std::string_view name,
                    double expected) {
  const auto* value = root.at(name);
  if (value == nullptr || !value->is_number() ||
      value->number() != expected) {
    return Status::InvalidArgument(std::string(name) +
                                   " does not match DeepSeek V4 Flash 0731");
  }
  return Status::Ok();
}

Status exact_string(const JsonValue& root, std::string_view name,
                    std::string_view expected) {
  const auto* value = root.at(name);
  if (value == nullptr || !value->is_string() ||
      value->string() != expected) {
    return Status::InvalidArgument(std::string(name) +
                                   " does not match DeepSeek V4 Flash 0731");
  }
  return Status::Ok();
}

Status exact_bool(const JsonValue& root, std::string_view name, bool expected) {
  const auto* value = root.at(name);
  if (value == nullptr || !value->is_boolean() ||
      value->boolean() != expected) {
    return Status::InvalidArgument(std::string(name) +
                                   " does not match DeepSeek V4 Flash 0731");
  }
  return Status::Ok();
}

Status absent(const JsonValue& root, std::string_view name) {
  if (root.at(name) != nullptr) {
    return Status::InvalidArgument(std::string(name) +
                                   " is unsupported for DeepSeek V4 Flash 0731");
  }
  return Status::Ok();
}

template <std::size_t Size>
Status exact_integer_array(const JsonValue& root, std::string_view name,
                           const std::array<std::int64_t, Size>& expected) {
  const auto* value = root.at(name);
  if (value == nullptr || !value->is_array() ||
      value->array().size() != Size) {
    return Status::InvalidArgument(std::string(name) +
                                   " does not match DeepSeek V4 Flash 0731");
  }
  for (std::size_t index = 0; index < Size; ++index) {
    if (!value->array()[index].is_integer() ||
        value->array()[index].integer() != expected[index]) {
      return Status::InvalidArgument(std::string(name) +
                                     " does not match DeepSeek V4 Flash 0731");
    }
  }
  return Status::Ok();
}

}  // namespace

Result<DeepSeekV4Config> DeepSeekV4Config::ParseFlash0731(
    std::string_view json) {
  JsonLimits limits;
  limits.max_input_bytes = kMaxConfigBytes;
  limits.max_depth = 16;
  limits.max_nodes = 4096;
  limits.max_string_bytes = 4096;
  auto root = JsonValue::Parse(json, limits);
  if (!root.ok()) return root.status();
  if (!root->is_object())
    return Status::InvalidArgument("DeepSeek config must be an object");

  const auto* architectures = root->at("architectures");
  if (architectures == nullptr || !architectures->is_array() ||
      architectures->array().size() != 1 ||
      !architectures->array()[0].is_string() ||
      architectures->array()[0].string() != "DeepseekV4ForCausalLM") {
    return Status::InvalidArgument(
        "architectures does not match DeepSeek V4 Flash 0731");
  }
  for (const auto& status : {
           exact_string(*root, "model_type", "deepseek_v4"),
           exact_string(*root, "torch_dtype", "bfloat16"),
           exact_string(*root, "hidden_act", "silu"),
           exact_string(*root, "expert_dtype", "fp4"),
           exact_string(*root, "scoring_func", "sqrtsoftplus"),
           exact_string(*root, "topk_method", "noaux_tc"),
           exact_bool(*root, "attention_bias", false),
           exact_bool(*root, "norm_topk_prob", true),
           exact_bool(*root, "tie_word_embeddings", false),
           exact_bool(*root, "use_cache", true),
           exact_integer(*root, "bos_token_id", 0),
           exact_integer(*root, "eos_token_id", 1),
           exact_integer(*root, "hidden_size", 4096),
           exact_integer(*root, "num_hidden_layers", 43),
           exact_integer(*root, "num_hash_layers", 3),
           exact_integer(*root, "num_nextn_predict_layers", 1),
           exact_integer(*root, "num_attention_heads", 64),
           exact_integer(*root, "num_key_value_heads", 1),
           exact_integer(*root, "head_dim", 512),
           exact_integer(*root, "qk_rope_head_dim", 64),
           exact_integer(*root, "n_routed_experts", 256),
           exact_integer(*root, "n_shared_experts", 1),
           exact_integer(*root, "num_experts_per_tok", 6),
           exact_integer(*root, "moe_intermediate_size", 2048),
           exact_integer(*root, "vocab_size", 129280),
           exact_integer(*root, "max_position_embeddings", 1048576),
           exact_integer(*root, "sliding_window", 128),
           exact_integer(*root, "index_n_heads", 64),
           exact_integer(*root, "index_head_dim", 128),
           exact_integer(*root, "index_topk", 512),
           exact_integer(*root, "q_lora_rank", 1024),
           exact_integer(*root, "o_lora_rank", 1024),
           exact_integer(*root, "o_groups", 8),
           exact_integer(*root, "hc_mult", 4),
           exact_integer(*root, "hc_sinkhorn_iters", 20),
           exact_integer(*root, "compress_rope_theta", 160000),
           exact_integer(*root, "dspark_block_size", 5),
           exact_integer(*root, "dspark_noise_token_id", 128799),
           exact_integer(*root, "dspark_markov_rank", 256),
           exact_number(*root, "attention_dropout", 0.0),
           exact_number(*root, "rms_norm_eps", 0.000001),
           exact_number(*root, "hc_eps", 0.000001),
           exact_number(*root, "rope_theta", 10000.0),
           exact_number(*root, "routed_scaling_factor", 1.5),
           exact_number(*root, "swiglu_limit", 10.0)}) {
    if (!status.ok()) return status;
  }

  const auto* quantization = root->at("quantization_config");
  if (quantization == nullptr || !quantization->is_object())
    return Status::InvalidArgument("DeepSeek quantization_config is invalid");
  for (const auto& status : {
           exact_string(*quantization, "activation_scheme", "dynamic"),
           exact_string(*quantization, "fmt", "e4m3"),
           exact_string(*quantization, "quant_method", "fp8"),
           exact_string(*quantization, "scale_fmt", "ue8m0"),
           exact_integer_array(*quantization, "weight_block_size",
                               std::array<std::int64_t, 2>{128, 128}),
           // The vLLM NVFP4 override selects its DeepGEMM MegaMoE path,
           // which itself requires Blackwell (SM100).  Flash-0731 V1 is an
           // SM89/SM90 MXFP4 compatibility profile, so admitting this
           // checkpoint switch would misdescribe the executable weights.
           absent(*quantization, "moe_quant_algo")}) {
    if (!status.ok()) return status;
  }
  const auto* rope = root->at("rope_scaling");
  if (rope == nullptr || !rope->is_object())
    return Status::InvalidArgument("DeepSeek rope_scaling is invalid");
  for (const auto& status : {
           exact_string(*rope, "type", "yarn"),
           exact_integer(*rope, "beta_fast", 32),
           exact_integer(*rope, "beta_slow", 1),
           exact_integer(*rope, "factor", 16),
           exact_integer(*rope, "original_max_position_embeddings", 65536)}) {
    if (!status.ok()) return status;
  }
  constexpr std::array<std::int64_t, 46> ratios{
      0,0,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,
      128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,0,0,0};
  auto status = exact_integer_array(*root, "compress_ratios", ratios);
  if (status.ok()) status = exact_integer_array(
      *root, "dspark_target_layer_ids",
      std::array<std::int64_t, 3>{40, 41, 42});
  if (!status.ok()) return status;
  return Flash0731();
}

Result<DeepSeekV4InferenceConfig>
DeepSeekV4InferenceConfig::ParseFlash0731(std::string_view json) {
  JsonLimits limits;
  limits.max_input_bytes = kMaxConfigBytes;
  limits.max_depth = 16;
  limits.max_nodes = 4096;
  limits.max_string_bytes = 4096;
  auto root = JsonValue::Parse(json, limits);
  if (!root.ok()) return root.status();
  if (!root->is_object()) {
    return Status::InvalidArgument(
        "DeepSeek inference config must be an object");
  }
  auto status = exact_integer(*root, "n_mtp_layers", 3);
  if (!status.ok()) return status;
  return Flash0731();
}

Result<DeepSeekV4ConfigFileReceipt>
load_deepseek_v4_flash_0731_config_file(
    const std::filesystem::path& path) {
  if (path.empty())
    return Status::InvalidArgument("DeepSeek config path is empty");
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) return Status::Unavailable("cannot open DeepSeek config");
  const auto end = stream.tellg();
  if (end <= 0 || static_cast<std::uint64_t>(end) >
                      DeepSeekV4Config::kMaxConfigBytes) {
    return Status::InvalidArgument("DeepSeek config file size is invalid");
  }
  std::string source(static_cast<std::size_t>(end), '\0');
  stream.seekg(0);
  if (!stream.read(source.data(), static_cast<std::streamsize>(source.size())))
    return Status::Unavailable("cannot read complete DeepSeek config");
  auto config = DeepSeekV4Config::ParseFlash0731(source);
  if (!config.ok()) return config.status();
  auto digest = sha256(std::as_bytes(std::span(source)));
  if (!digest.ok()) return digest.status();
  return DeepSeekV4ConfigFileReceipt{
      *config, static_cast<std::uint64_t>(source.size()), *digest};
}

Result<DeepSeekV4InferenceConfigFileReceipt>
load_deepseek_v4_flash_0731_inference_config_file(
    const std::filesystem::path& path) {
  if (path.empty()) {
    return Status::InvalidArgument("DeepSeek inference config path is empty");
  }
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) {
    return Status::Unavailable("cannot open DeepSeek inference config");
  }
  const auto end = stream.tellg();
  if (end <= 0 || static_cast<std::uint64_t>(end) >
                      DeepSeekV4InferenceConfig::kMaxConfigBytes) {
    return Status::InvalidArgument(
        "DeepSeek inference config file size is invalid");
  }
  std::string source(static_cast<std::size_t>(end), '\0');
  stream.seekg(0);
  if (!stream.read(source.data(), static_cast<std::streamsize>(source.size()))) {
    return Status::Unavailable("cannot read complete DeepSeek inference config");
  }
  auto config = DeepSeekV4InferenceConfig::ParseFlash0731(source);
  if (!config.ok()) return config.status();
  auto digest = sha256(std::as_bytes(std::span(source)));
  if (!digest.ok()) return digest.status();
  return DeepSeekV4InferenceConfigFileReceipt{
      *config, static_cast<std::uint64_t>(source.size()), *digest};
}

Status DeepSeekV4Config::validate() const {
  if (main_layers != 43 || nextn_predict_layers != 1 || hidden_size != 4096 ||
      hc_streams != 4 || attention_heads != 64 || routed_experts != 256 ||
      activated_experts != 6 || vocabulary_size != 129280 ||
      maximum_positions != 1048576 || rms_norm_epsilon != 0.000001) {
    return Status::InvalidArgument(
        "DeepSeek V4 Flash 0731 geometry does not match the pinned revision");
  }
  return Status::Ok();
}

Status DeepSeekV4InferenceConfig::validate() const {
  if (mtp_stages != 3) {
    return Status::InvalidArgument(
        "DeepSeek V4 Flash 0731 inference MTP stage count does not match the pinned revision");
  }
  return Status::Ok();
}

Result<DeepSeekPipelinePlan> DeepSeekPipelinePlan::Create(
    std::uint32_t world_size, bool dspark_enabled) {
  if (world_size < 1 || world_size > 4) {
    return Status::InvalidArgument("DeepSeek pipeline world size must be 1..4");
  }
  static const std::array<std::vector<DeepSeekStageRange>, 4> kEnabled = {{
      {{0, 42}},
      {{0, 22}, {23, 42}},
      {{0, 14}, {15, 30}, {31, 42}},
      {{0, 10}, {11, 22}, {23, 34}, {35, 42}},
  }};
  static const std::array<std::vector<DeepSeekStageRange>, 4> kDisabled = {{
      {{0, 42}},
      {{0, 21}, {22, 42}},
      {{0, 13}, {14, 28}, {29, 42}},
      {{0, 10}, {11, 21}, {22, 32}, {33, 42}},
  }};
  DeepSeekPipelinePlan plan;
  plan.ranges_ = (dspark_enabled ? kEnabled : kDisabled)[world_size - 1];
  plan.stages_.reserve(world_size);
  for (std::uint32_t rank = 0; rank < world_size; ++rank) {
    plan.stages_.push_back({rank, plan.ranges_[rank], rank == 0,
                            rank + 1 == world_size,
                            dspark_enabled && rank + 1 == world_size});
  }
  return plan;
}

const DeepSeekStagePlan& DeepSeekPipelinePlan::rank(std::uint32_t rank) const {
  if (rank >= stages_.size()) throw std::out_of_range("DeepSeek pipeline rank");
  return stages_[rank];
}

}  // namespace pih
