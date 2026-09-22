#include "pih/model/qwen3_manifest.h"

#include <string>
#include <utility>

namespace pih {
namespace {

std::vector<ExpectedTensor> build_expected_tensors() {
  std::vector<ExpectedTensor> tensors;
  tensors.reserve(Qwen3Manifest::kOfficialTensorCount);
  tensors.push_back({"lm_head.weight", {151936, 1024}});
  tensors.push_back({"model.embed_tokens.weight", {151936, 1024}});
  for (std::uint64_t layer = 0; layer < 28; ++layer) {
    const auto prefix = "model.layers." + std::to_string(layer) + ".";
    tensors.push_back({prefix + "input_layernorm.weight", {1024}});
    tensors.push_back({prefix + "mlp.down_proj.weight", {1024, 3072}});
    tensors.push_back({prefix + "mlp.gate_proj.weight", {3072, 1024}});
    tensors.push_back({prefix + "mlp.up_proj.weight", {3072, 1024}});
    tensors.push_back({prefix + "post_attention_layernorm.weight", {1024}});
    tensors.push_back({prefix + "self_attn.k_norm.weight", {128}});
    tensors.push_back({prefix + "self_attn.k_proj.weight", {1024, 1024}});
    tensors.push_back({prefix + "self_attn.o_proj.weight", {1024, 2048}});
    tensors.push_back({prefix + "self_attn.q_norm.weight", {128}});
    tensors.push_back({prefix + "self_attn.q_proj.weight", {2048, 1024}});
    tensors.push_back({prefix + "self_attn.v_proj.weight", {1024, 1024}});
  }
  tensors.push_back({"model.norm.weight", {1024}});
  return tensors;
}

}  // namespace

const std::vector<ExpectedTensor>& Qwen3Manifest::expected_tensors() {
  static const auto tensors = build_expected_tensors();
  return tensors;
}

Status Qwen3Manifest::Validate(const SafetensorsHeader& header) {
  const auto& expected = expected_tensors();
  if (expected.size() != kOfficialTensorCount ||
      header.tensors().size() != kOfficialTensorCount) {
    return Status::InvalidArgument("Qwen3 source tensor count does not match manifest");
  }
  if (header.data_bytes() != kOfficialSourcePayloadBytes) {
    return Status::InvalidArgument("Qwen3 source payload bytes do not match manifest");
  }
  for (const auto& requirement : expected) {
    const auto* record = header.tensor(requirement.name);
    if (record == nullptr) {
      return Status::InvalidArgument("Qwen3 source tensor is missing: " +
                                     requirement.name);
    }
    if (record->dtype != DType::kBFloat16 || record->shape != requirement.shape) {
      return Status::InvalidArgument("Qwen3 source tensor contract mismatch: " +
                                     requirement.name);
    }
  }
  return Status::Ok();
}

}  // namespace pih
