#include "pih/model/deepseek_rank_process_manifest_plan.h"

#include <limits>

namespace pih {

Result<DeepSeekRankProcessManifestPlan>
DeepSeekRankProcessManifestPlan::Create(
    std::uint64_t engine_epoch, std::uint64_t worker_generation,
    std::uint64_t first_process_manifest_identity,
    std::uint64_t startup_deadline_ns,
    const DeepSeekPhysicalDeviceRegistry& devices) {
  const auto world_size = devices.world_size();
  if (engine_epoch == 0 || worker_generation == 0 || world_size == 0 ||
      world_size > 4 || first_process_manifest_identity == 0 ||
      startup_deadline_ns == 0 ||
      first_process_manifest_identity >
          std::numeric_limits<std::uint64_t>::max() - (world_size - 1U)) {
    return Status::InvalidArgument(
        "DeepSeek rank process manifest plan identity is invalid");
  }
  std::vector<DeepSeekRankProcessManifest> manifests;
  manifests.reserve(world_size);
  for (std::uint32_t rank = 0; rank < world_size; ++rank) {
    const auto* device = devices.rank(rank);
    if (device == nullptr)
      return Status::Internal("DeepSeek physical device registry has a rank gap");
    manifests.push_back({engine_epoch, worker_generation, world_size, rank,
                         device->registry_identity,
                         first_process_manifest_identity + rank,
                         device->uuid_commitment,
                         device->startup_device_ordinal,
                         startup_deadline_ns});
  }
  return DeepSeekRankProcessManifestPlan(std::move(manifests));
}

}  // namespace pih
