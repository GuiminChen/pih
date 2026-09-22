#include "pih/model/deepseek_optimization_policy.h"

#include "pih/core/canonical_hash.h"

namespace pih {

Result<DeepSeekOptimizationPolicy> DeepSeekOptimizationPolicy::Create(
    DeepSeekOptimizationProfile profile) {
  if ((profile.architecture != DeepSeekGpuArchitecture::kSm89 &&
       profile.architecture != DeepSeekGpuArchitecture::kSm90) ||
      (profile.attention_layout !=
           DeepSeekAttentionPhysicalLayoutVersion::kBaseline &&
       profile.attention_layout != DeepSeekAttentionPhysicalLayoutVersion::kV1) ||
      (profile.fused_kernel_set != DeepSeekFusedKernelSet::kUnfusedControl &&
       profile.fused_kernel_set != DeepSeekFusedKernelSet::kVerifiedV1)) {
    return Status::InvalidArgument(
        "DeepSeek optimization profile contains an unknown closed value");
  }
  if (profile.world_size == 0 || profile.world_size > 4) {
    return Status::InvalidArgument(
        "DeepSeek optimization world size is outside PP1-PP4");
  }
  if (profile.cuda_graph) {
    return Status::FailedPrecondition(
        "DeepSeek V1 forbids CUDA Graph optimization");
  }
  if (profile.expert_locality_reorder) {
    return Status::FailedPrecondition(
        "DeepSeek V1 forbids expert-locality reordering");
  }
  if (profile.dspark &&
      profile.architecture != DeepSeekGpuArchitecture::kSm90) {
    return Status::FailedPrecondition(
        "DeepSeek DSpark production is restricted to SM90");
  }
  auto hash = CanonicalHashBuilder::Create(
      "pih:deepseek-optimization-policy:v1", 9);
  if (!hash.ok()) return hash.status();
  const std::uint32_t architecture =
      profile.architecture == DeepSeekGpuArchitecture::kSm89 ? 89U : 90U;
  Status status = hash->add_u32(1, architecture);
  if (status.ok()) status = hash->add_u32(2, profile.world_size);
  if (status.ok()) status = hash->add_u32(3, profile.chunked_prefill);
  if (status.ok()) {
    status = hash->add_u32(4, static_cast<std::uint32_t>(profile.attention_layout));
  }
  if (status.ok()) {
    status = hash->add_u32(5, static_cast<std::uint32_t>(profile.fused_kernel_set));
  }
  if (status.ok()) status = hash->add_u32(6, profile.observed_miss_debit);
  if (status.ok()) status = hash->add_u32(7, profile.dspark);
  if (status.ok()) status = hash->add_u32(8, profile.cuda_graph);
  if (status.ok()) status = hash->add_u32(9, profile.expert_locality_reorder);
  if (!status.ok()) return status;
  auto identity = hash->finalize();
  if (!identity.ok()) return identity.status();
  return DeepSeekOptimizationPolicy(profile, *identity);
}

}  // namespace pih
