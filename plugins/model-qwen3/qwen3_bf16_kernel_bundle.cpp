#include "pih/model/qwen3_bf16_kernel_bundle.h"

#include <array>
#include <string>

#include "pih/backend/cuda/cubin_artifact_manifest.h"
#include "qwen_cubin_identity.h"

namespace pih {
namespace {

constexpr std::array<QwenBf16Primitive, QwenBf16KernelBundle::kPrimitiveCount>
    kPrimitives{QwenBf16Primitive::kEmbedding,
                QwenBf16Primitive::kResidualAdd,
                QwenBf16Primitive::kSiluMul,
                QwenBf16Primitive::kRmsNorm,
                QwenBf16Primitive::kRope,
                QwenBf16Primitive::kKvAppend,
                QwenBf16Primitive::kPagedGqa,
                QwenBf16Primitive::kRopeAngles,
                QwenBf16Primitive::kGreedyArgmax,
                QwenBf16Primitive::kTeacherForcedMetric};

constexpr std::array<QwenBf16PackedPrimitive,
                     QwenBf16KernelBundle::kPackedPrimitiveCount>
    kPackedPrimitives{QwenBf16PackedPrimitive::kEmbedding,
                      QwenBf16PackedPrimitive::kRopeAngles,
                      QwenBf16PackedPrimitive::kKvAppend,
                      QwenBf16PackedPrimitive::kPagedGqa,
                      QwenBf16PackedPrimitive::kSampleHidden,
                      QwenBf16PackedPrimitive::kGreedyArgmax,
                      QwenBf16PackedPrimitive::kSampler};

Result<std::uint32_t> exact_target_sm(std::uint32_t major,
                                      std::uint32_t minor) {
  if (major == 8 && minor == 9) return UINT32_C(89);
  if (major == 9 && minor == 0) return UINT32_C(90);
  return Status::InvalidArgument(
      "Qwen BF16 V1 supports only compute capability 8.9 or 9.0");
}

Result<std::size_t> primitive_index(QwenBf16Primitive primitive) {
  const auto index = static_cast<std::size_t>(primitive);
  if (index >= kPrimitives.size() || kPrimitives[index] != primitive) {
    return Status::InvalidArgument("unknown Qwen BF16 primitive");
  }
  return index;
}

Result<std::size_t> packed_primitive_index(QwenBf16PackedPrimitive primitive) {
  for (std::size_t index = 0; index < kPackedPrimitives.size(); ++index) {
    if (kPackedPrimitives[index] == primitive) return index;
  }
  return Status::InvalidArgument("unknown packed Qwen BF16 primitive");
}

}  // namespace

Result<QwenBf16KernelBundle> QwenBf16KernelBundle::Load(
    KernelModuleDriver& driver, const std::filesystem::path& manifest_path,
    const std::filesystem::path& cubin_path, std::uint32_t compute_major,
    std::uint32_t compute_minor, std::uint64_t maximum_cubin_bytes) {
  auto expected_sm = exact_target_sm(compute_major, compute_minor);
  if (!expected_sm.ok()) return expected_sm.status();
  auto artifact = CubinArtifactManifest::Load(manifest_path);
  if (!artifact.ok()) return artifact.status();
  if (artifact->target_sm() != expected_sm.value()) {
    return Status::InvalidArgument(
        "cubin target does not match the selected NVIDIA device");
  }
  auto cubin = artifact->load_cubin(cubin_path, maximum_cubin_bytes);
  if (!cubin.ok()) return cubin.status();

  const std::string digest = cubin->digest().hex();
  // Compare the same owned snapshot subsequently passed to the CUDA driver.
  // A replaced cubin + self-consistent sidecar cannot change the identity
  // compiled into the authenticated model DSO.
  if (digest != pih_expected_qwen_cubin_sha256(expected_sm.value())) {
    return Status::FailedPrecondition("Qwen cubin differs from authenticated build");
  }
  std::vector<KernelSignatureManifest> manifests;
  manifests.reserve(kFunctionCount);
  for (std::size_t index = 0; index < kPrimitives.size(); ++index) {
    auto manifest = qwen_bf16_kernel_manifest(kPrimitives[index], digest);
    if (!manifest.ok()) return manifest.status();
    manifests.push_back(std::move(manifest).value());
  }
  for (const auto primitive : kPackedPrimitives) {
    auto manifest = qwen_bf16_packed_kernel_manifest(primitive, digest);
    if (!manifest.ok()) return manifest.status();
    manifests.push_back(std::move(manifest).value());
  }
  std::array<KernelEntrypointRequest, kFunctionCount> requests;
  for (std::size_t index = 0; index < kPrimitives.size(); ++index) {
    auto symbol = qwen_bf16_kernel_symbol(kPrimitives[index]);
    if (!symbol.ok()) return symbol.status();
    requests[index] = {symbol.value(), &manifests[index]};
  }
  for (std::size_t index = 0; index < kPackedPrimitives.size(); ++index) {
    auto symbol = qwen_bf16_packed_kernel_symbol(kPackedPrimitives[index]);
    if (!symbol.ok()) return symbol.status();
    const auto request_index = kPrimitiveCount + index;
    requests[request_index] = {symbol.value(), &manifests[request_index]};
  }
  auto module = VerifiedKernelModule::Load(driver, cubin.value(), requests);
  if (!module.ok()) return module.status();
  return QwenBf16KernelBundle(expected_sm.value(), std::move(manifests),
                              std::move(module).value());
}

Result<const ResolvedKernelFunction*> QwenBf16KernelBundle::packed_function(
    QwenBf16PackedPrimitive primitive) const {
  auto index = packed_primitive_index(primitive);
  if (!index.ok()) return index.status();
  return &module_.functions()[kPrimitiveCount + index.value()];
}

Result<const ResolvedKernelFunction*> QwenBf16KernelBundle::function(
    QwenBf16Primitive primitive) const {
  auto index = primitive_index(primitive);
  if (!index.ok()) return index.status();
  return &module_.functions()[index.value()];
}

}  // namespace pih
