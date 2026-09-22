#include "pih/model/qwen3_numerical_run.h"

#include <array>
#include <cstddef>
#include <string>

#include "pih/core/bounded_json.h"

namespace pih {
namespace {

constexpr std::array<std::string_view, 10> kFields{
    "schema", "model_sha256", "fixture_sha256", "tolerance_sha256",
    "kernel_bundle_sha256", "build_sha256", "environment_sha256",
    "target_gpu", "target_sm", "run_generation"};

bool known_field(std::string_view name) {
  for (const auto field : kFields) if (field == name) return true;
  return false;
}

Result<std::string> string_field(const JsonValue& root, std::string_view name) {
  const auto* value = root.at(name);
  if (value == nullptr || !value->is_string() || value->string().empty()) {
    return Status::InvalidArgument(std::string(name) + " is invalid");
  }
  return value->string();
}

Result<Sha256Digest> digest_field(const JsonValue& root, std::string_view name) {
  auto text = string_field(root, name);
  if (!text.ok()) return text.status();
  return Sha256Digest::ParseHex(text.value());
}

}  // namespace

Result<QwenNumericalRunIdentity> QwenNumericalRunIdentity::Parse(
    std::string_view json) {
  JsonLimits limits;
  limits.max_input_bytes = kMaximumManifestBytes;
  limits.max_depth = 3;
  limits.max_nodes = 24;
  limits.max_string_bytes = 96;
  auto root = JsonValue::Parse(json, limits);
  if (!root.ok()) return root.status();
  if (!root->is_object() || root->object().size() != kFields.size()) {
    return Status::InvalidArgument("Qwen numerical run fields are invalid");
  }
  for (const auto& [name, value] : root->object()) {
    (void)value;
    if (!known_field(name)) {
      return Status::InvalidArgument("unknown Qwen numerical run field");
    }
  }
  auto schema = string_field(*root, "schema");
  auto gpu = string_field(*root, "target_gpu");
  auto sm = string_field(*root, "target_sm");
  auto model = digest_field(*root, "model_sha256");
  auto fixture = digest_field(*root, "fixture_sha256");
  auto tolerance = digest_field(*root, "tolerance_sha256");
  auto kernels = digest_field(*root, "kernel_bundle_sha256");
  auto build = digest_field(*root, "build_sha256");
  auto environment = digest_field(*root, "environment_sha256");
  if (!schema.ok()) return schema.status();
  if (!gpu.ok()) return gpu.status();
  if (!sm.ok()) return sm.status();
  if (!model.ok()) return model.status();
  if (!fixture.ok()) return fixture.status();
  if (!tolerance.ok()) return tolerance.status();
  if (!kernels.ok()) return kernels.status();
  if (!build.ok()) return build.status();
  if (!environment.ok()) return environment.status();
  if (schema.value() != "pih.qwen_bf16_numerical_run.v1") {
    return Status::InvalidArgument("unsupported Qwen numerical run schema");
  }
  QwenTargetGpu target_gpu;
  std::uint32_t target_sm;
  if (gpu.value() == "RTX_4090_D" && sm.value() == "sm_89") {
    target_gpu = QwenTargetGpu::kRtx4090D;
    target_sm = 89;
  } else if (gpu.value() == "H100_PCIE_80GB" && sm.value() == "sm_90") {
    target_gpu = QwenTargetGpu::kH100Pcie80Gb;
    target_sm = 90;
  } else {
    return Status::InvalidArgument("Qwen numerical GPU and SM mismatch");
  }
  const auto* generation = root->at("run_generation");
  if (generation == nullptr || !generation->is_integer() ||
      generation->integer() <= 0) {
    return Status::InvalidArgument("Qwen numerical run generation is invalid");
  }
  return QwenNumericalRunIdentity(
      model.value(), fixture.value(), tolerance.value(), kernels.value(),
      build.value(), environment.value(), target_gpu, target_sm,
      static_cast<std::uint64_t>(generation->integer()));
}

Result<QwenNumericalReport> compare_qwen_numerical_run(
    const QwenNumericalRunIdentity& reference_identity,
    std::span<const float> reference,
    const QwenNumericalRunIdentity& candidate_identity,
    std::span<const float> candidate, const QwenNumericalPolicy& policy) {
  if (!(reference_identity == candidate_identity)) {
    return Status::FailedPrecondition(
        "Qwen numerical run identities cannot be spliced");
  }
  return compare_qwen_numerics(reference, candidate, policy);
}

Result<Sha256Digest> QwenNumericalRunIdentity::semantic_digest() const {
  const std::string canonical =
      "pih.qwen_bf16_numerical_run.identity.v1\n" + model_.hex() + "\n" +
      fixture_.hex() + "\n" + tolerance_.hex() + "\n" + kernels_.hex() +
      "\n" + build_.hex() + "\n" + environment_.hex() + "\n" +
      std::to_string(static_cast<std::uint8_t>(target_gpu_)) + "\n" +
      std::to_string(target_sm_) + "\n" + std::to_string(run_generation_);
  return sha256(std::as_bytes(std::span(canonical.data(), canonical.size())));
}

}  // namespace pih
