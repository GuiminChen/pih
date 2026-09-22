#include "pih/model/deepseek_fused_kernel_plan.h"

#include "pih/core/canonical_hash.h"

namespace pih {
namespace {

constexpr std::string_view kFusedLogicalId =
    "deepseek.flash0731.fused.v1";
constexpr std::string_view kUnfusedLogicalId =
    "deepseek.flash0731.unfused.control.v1";

bool nonzero(const Sha256Digest& digest) {
  for (const auto value : digest.bytes) {
    if (value != std::byte{0}) return true;
  }
  return false;
}

}  // namespace

Result<Sha256Digest> DeepSeekFusedKernelPlan::ExpectedAbiRoot(
    DeepSeekGpuArchitecture architecture) {
  if (architecture != DeepSeekGpuArchitecture::kSm89 &&
      architecture != DeepSeekGpuArchitecture::kSm90) {
    return Status::InvalidArgument(
        "DeepSeek fused kernel architecture is unknown");
  }
  // This is the offline compiler ABI manifest for
  // deepseek_fused_residual_rms_norm_kernel. Offsets describe the packed
  // driver parameter buffer and deliberately include the architecture so an
  // SM89 artifact cannot be relabelled as SM90 (or vice versa).
  constexpr std::uint32_t kOffsets[] = {0, 8, 16, 24, 32, 40, 48};
  constexpr std::uint32_t kWidths[] = {8, 8, 8, 8, 8, 8, 4};
  auto hash = CanonicalHashBuilder::Create(
      "pih:deepseek-fused-residual-rmsnorm-abi:v1", 20);
  if (!hash.ok()) return hash.status();
  Status status = hash->add_ascii_utf8(
      1, "deepseek_fused_residual_rms_norm_kernel");
  if (status.ok()) {
    status = hash->add_u32(2, static_cast<std::uint32_t>(architecture));
  }
  if (status.ok()) status = hash->add_u32(3, 7);
  for (std::uint32_t index = 0; index < 7 && status.ok(); ++index) {
    status = hash->add_u32(4 + index * 2, kOffsets[index]);
    if (status.ok()) status = hash->add_u32(5 + index * 2, kWidths[index]);
  }
  if (status.ok()) status = hash->add_u32(18, 4096);
  if (status.ok()) status = hash->add_u32(19, 256);
  if (status.ok()) status = hash->add_ascii_utf8(20, "bf16-fp32-bf16");
  if (!status.ok()) return status;
  return hash->finalize();
}

Result<DeepSeekFusedKernelPlan> DeepSeekFusedKernelPlan::Compile(
    const DeepSeekOptimizationPolicy& policy, std::uint32_t token_count,
    const DeepSeekFusedKernelManifest* manifest,
    const Sha256Digest* observed_cubin_digest) {
  if (token_count == 0 || token_count > 4096) {
    return Status::InvalidArgument(
        "DeepSeek fused kernel token shape is unsupported");
  }
  DeepSeekFusedKernelPlan result;
  Sha256Digest abi{};
  if (policy.fused_kernel_set() == DeepSeekFusedKernelSet::kUnfusedControl) {
    if (manifest != nullptr || observed_cubin_digest != nullptr) {
      return Status::InvalidArgument(
          "DeepSeek unfused control forbids a fused manifest");
    }
    result.logical_id_ = kUnfusedLogicalId;
  } else {
    if (manifest == nullptr || observed_cubin_digest == nullptr) {
      return Status::InvalidArgument(
          "DeepSeek verified fused selection requires a sealed manifest");
    }
    auto expected_abi = ExpectedAbiRoot(policy.architecture());
    if (!expected_abi.ok()) return expected_abi.status();
    if (manifest->logical_id != kFusedLogicalId ||
        manifest->architecture != policy.architecture() ||
        manifest->maximum_tokens != 4096 || manifest->hidden_size != 4096 ||
        manifest->routed_experts != 256 || manifest->activated_experts != 6 ||
        token_count > manifest->maximum_tokens ||
        !nonzero(manifest->cubin_digest) ||
        manifest->cubin_digest != *observed_cubin_digest ||
        manifest->parameter_abi_root != *expected_abi) {
      return Status::InvalidArgument(
          "DeepSeek fused kernel manifest, artifact, shape, or ABI drift");
    }
    result.fused_ = true;
    result.logical_id_ = manifest->logical_id;
    result.cubin_digest_ = manifest->cubin_digest;
    abi = manifest->parameter_abi_root;
  }
  auto hash = CanonicalHashBuilder::Create(
      "pih:deepseek-fused-kernel-plan:v1", result.fused_ ? 7 : 5);
  if (!hash.ok()) return hash.status();
  Status status = hash->add_hash(1, policy.identity());
  if (status.ok()) status = hash->add_u32(2, token_count);
  if (status.ok()) status = hash->add_u32(3, result.fused_);
  if (status.ok()) status = hash->add_ascii_utf8(4, result.logical_id_);
  if (status.ok()) {
    status = hash->add_u32(5, static_cast<std::uint32_t>(policy.architecture()));
  }
  if (result.fused_ && status.ok()) {
    status = hash->add_hash(6, result.cubin_digest_);
  }
  if (result.fused_ && status.ok()) status = hash->add_hash(7, abi);
  if (!status.ok()) return status;
  auto identity = hash->finalize();
  if (!identity.ok()) return identity.status();
  result.identity_ = *identity;
  return result;
}

}  // namespace pih
