#include "pih/model/qwen3_config.h"

#include <string>

#include "pih/core/bounded_json.h"

namespace pih {
namespace {

Result<std::uint64_t> required_u64(const JsonValue& root, std::string_view name) {
  const auto* value = root.at(name);
  if (value == nullptr || !value->is_integer() || value->integer() < 0) {
    return Status::InvalidArgument(std::string(name) + " must be a nonnegative integer");
  }
  return static_cast<std::uint64_t>(value->integer());
}

Result<std::int64_t> required_i64(const JsonValue& root, std::string_view name) {
  const auto* value = root.at(name);
  if (value == nullptr || !value->is_integer()) {
    return Status::InvalidArgument(std::string(name) + " must be an integer");
  }
  return value->integer();
}

Status require_string(const JsonValue& root, std::string_view name,
                      std::string_view expected) {
  const auto* value = root.at(name);
  if (value == nullptr || !value->is_string() || value->string() != expected) {
    return Status::InvalidArgument(std::string(name) + " does not match Qwen3-0.6B");
  }
  return Status::Ok();
}

Status require_bool(const JsonValue& root, std::string_view name, bool expected) {
  const auto* value = root.at(name);
  if (value == nullptr || !value->is_boolean() || value->boolean() != expected) {
    return Status::InvalidArgument(std::string(name) + " does not match Qwen3-0.6B");
  }
  return Status::Ok();
}

template <typename T>
Status require_exact(T actual, T expected, std::string_view name) {
  if (actual != expected) {
    return Status::InvalidArgument(std::string(name) + " does not match Qwen3-0.6B");
  }
  return Status::Ok();
}

}  // namespace

Result<Qwen3Config> Qwen3Config::Parse(std::string_view json) {
  JsonLimits limits;
  limits.max_input_bytes = kMaxConfigBytes;
  limits.max_depth = 16;
  limits.max_nodes = 4096;
  limits.max_string_bytes = 4096;
  auto root = JsonValue::Parse(json, limits);
  if (!root.ok()) return root.status();
  if (!root->is_object()) return Status::InvalidArgument("Qwen config must be an object");

  // The native kernels implement full attention, unscaled RoPE and bias-free
  // projections. Do not authenticate a config whose semantics we then ignore.
  for (const auto name : {"attention_bias", "mlp_bias", "use_sliding_window"}) {
    const auto* value = root->at(name);
    if (value != nullptr && (!value->is_boolean() || value->boolean())) {
      return Status::InvalidArgument(std::string(name) + " is unsupported by native Qwen");
    }
  }
  if (const auto* scaling = root->at("rope_scaling");
      scaling != nullptr && !scaling->is_null()) {
    return Status::InvalidArgument("rope_scaling is unsupported by native Qwen");
  }

  for (const auto& requirement : {
           require_string(*root, "model_type", "qwen3"),
           require_string(*root, "torch_dtype", "bfloat16"),
           require_string(*root, "hidden_act", "silu"),
           require_bool(*root, "tie_word_embeddings", true),
           require_bool(*root, "use_cache", true)}) {
    if (!requirement.ok()) return requirement;
  }
  const auto* architectures = root->at("architectures");
  if (architectures == nullptr || !architectures->is_array() ||
      architectures->array().size() != 1 ||
      !architectures->array()[0].is_string() ||
      architectures->array()[0].string() != "Qwen3ForCausalLM") {
    return Status::InvalidArgument("architectures does not match Qwen3-0.6B");
  }

  auto hidden = required_u64(*root, "hidden_size");
  auto intermediate = required_u64(*root, "intermediate_size");
  auto layers = required_u64(*root, "num_hidden_layers");
  auto heads = required_u64(*root, "num_attention_heads");
  auto kv_heads = required_u64(*root, "num_key_value_heads");
  auto head_dim = required_u64(*root, "head_dim");
  auto vocabulary = required_u64(*root, "vocab_size");
  auto positions = required_u64(*root, "max_position_embeddings");
  auto bos = required_i64(*root, "bos_token_id");
  auto eos = required_i64(*root, "eos_token_id");
  for (const auto* result : {&hidden, &intermediate, &layers, &heads, &kv_heads,
                             &head_dim, &vocabulary, &positions}) {
    if (!result->ok()) return result->status();
  }
  if (!bos.ok()) return bos.status();
  if (!eos.ok()) return eos.status();
  const auto* theta = root->at("rope_theta");
  const auto* epsilon = root->at("rms_norm_eps");
  if (theta == nullptr || !theta->is_number() || epsilon == nullptr ||
      !epsilon->is_number()) {
    return Status::InvalidArgument("Qwen floating configuration fields are missing");
  }

  for (const auto& requirement : {
           require_exact(hidden.value(), UINT64_C(1024), "hidden_size"),
           require_exact(intermediate.value(), UINT64_C(3072), "intermediate_size"),
           require_exact(layers.value(), UINT64_C(28), "num_hidden_layers"),
           require_exact(heads.value(), UINT64_C(16), "num_attention_heads"),
           require_exact(kv_heads.value(), UINT64_C(8), "num_key_value_heads"),
           require_exact(head_dim.value(), UINT64_C(128), "head_dim"),
           require_exact(vocabulary.value(), UINT64_C(151936), "vocab_size"),
           require_exact(positions.value(), UINT64_C(40960), "max_position_embeddings"),
           require_exact(theta->number(), 1'000'000.0, "rope_theta"),
           require_exact(epsilon->number(), 0.000001, "rms_norm_eps"),
           require_exact(bos.value(), INT64_C(151643), "bos_token_id"),
           require_exact(eos.value(), INT64_C(151645), "eos_token_id")}) {
    if (!requirement.ok()) return requirement;
  }

  return Qwen3Config{hidden.value(), intermediate.value(), layers.value(),
                     heads.value(), kv_heads.value(), head_dim.value(),
                     vocabulary.value(), positions.value(), theta->number(),
                     epsilon->number(), bos.value(), eos.value()};
}

}  // namespace pih
