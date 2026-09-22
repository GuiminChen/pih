#include "pih/model/qwen3_int4_kernel_bundle.h"

#include <array>

#include "pih/backend/cuda/cubin_artifact_manifest.h"
#include "qwen_cubin_identity.h"

namespace pih {
namespace {

constexpr std::array<QwenBf16Primitive,
                     QwenInt4KernelBundle::kBf16PrimitiveCount>
    kBf16Primitives{QwenBf16Primitive::kEmbedding,
                    QwenBf16Primitive::kResidualAdd,
                    QwenBf16Primitive::kSiluMul,
                    QwenBf16Primitive::kRmsNorm,
                    QwenBf16Primitive::kRope,
                    QwenBf16Primitive::kKvAppend,
                    QwenBf16Primitive::kPagedGqa,
                    QwenBf16Primitive::kRopeAngles,
                    QwenBf16Primitive::kGreedyArgmax};

Result<std::uint32_t> exact_int4_target_sm(std::uint32_t major,
                                           std::uint32_t minor) {
  if (major == 8 && minor == 9) return UINT32_C(89);
  if (major == 9 && minor == 0) return UINT32_C(90);
  return Status::InvalidArgument(
      "Qwen INT4 V1 supports only compute capability 8.9 or 9.0");
}

}  // namespace

Result<QwenInt4KernelBundle> QwenInt4KernelBundle::Load(
    KernelModuleDriver& driver, const std::filesystem::path& manifest_path,
    const std::filesystem::path& cubin_path,
    std::uint32_t compute_major, std::uint32_t compute_minor,
    std::uint64_t maximum_cubin_bytes) {
  auto expected_sm = exact_int4_target_sm(compute_major, compute_minor);
  if (!expected_sm.ok()) return expected_sm.status();
  auto artifact = CubinArtifactManifest::Load(manifest_path);
  if (!artifact.ok()) return artifact.status();
  if (artifact->target_sm() != *expected_sm) {
    return Status::InvalidArgument(
        "Qwen INT4 cubin target does not match selected device");
  }
  auto cubin = artifact->load_cubin(cubin_path, maximum_cubin_bytes);
  if (!cubin.ok()) return cubin.status();
  if (cubin->digest().hex() != pih_expected_qwen_cubin_sha256(*expected_sm)) {
    return Status::FailedPrecondition("Qwen INT4 cubin differs from authenticated build");
  }
  auto plan = QwenInt4GemmPlan::Create(
      QwenInt4LinearShapeFamily::kQProj, 1);
  if (!plan.ok()) return plan.status();
  auto manifest = plan->signature(cubin->digest().hex());
  if (!manifest.ok()) return manifest.status();
  std::vector<KernelSignatureManifest> bf16_manifests;
  bf16_manifests.reserve(kBf16Primitives.size());
  for (const auto primitive : kBf16Primitives) {
    auto bf16_manifest =
        qwen_bf16_kernel_manifest(primitive, cubin->digest().hex());
    if (!bf16_manifest.ok()) return bf16_manifest.status();
    bf16_manifests.push_back(std::move(*bf16_manifest));
  }
  std::array<KernelEntrypointRequest, kFunctionCount> requests;
  for (std::size_t index = 0; index < kBf16Primitives.size(); ++index) {
    auto symbol = qwen_bf16_kernel_symbol(kBf16Primitives[index]);
    if (!symbol.ok()) return symbol.status();
    requests[index] = {symbol.value(), &bf16_manifests[index]};
  }
  requests.back() = {plan->kernel_symbol(), &*manifest};
  auto module = VerifiedKernelModule::Load(driver, *cubin, requests);
  if (!module.ok()) return module.status();
  return QwenInt4KernelBundle(*expected_sm, std::move(bf16_manifests),
                              std::move(*manifest), std::move(*module));
}

Result<const ResolvedKernelFunction*> QwenInt4KernelBundle::bf16_function(
    QwenBf16Primitive primitive) const {
  const auto index = static_cast<std::size_t>(primitive);
  if (index >= kBf16Primitives.size() || kBf16Primitives[index] != primitive) {
    return Status::InvalidArgument("unknown Qwen BF16 primitive");
  }
  return &module_.functions()[index];
}

}  // namespace pih
