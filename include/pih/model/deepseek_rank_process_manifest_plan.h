#pragma once

#include "pih/model/deepseek_physical_device_registry.h"
#include "pih/model/deepseek_rank_process_supervisor.h"

namespace pih {

class DeepSeekRankProcessManifestPlan final {
 public:
  static Result<DeepSeekRankProcessManifestPlan> Create(
      std::uint64_t engine_epoch, std::uint64_t worker_generation,
      std::uint64_t first_process_manifest_identity,
      std::uint64_t startup_deadline_ns,
      const DeepSeekPhysicalDeviceRegistry& devices);
  [[nodiscard]] std::span<const DeepSeekRankProcessManifest> manifests()
      const noexcept { return manifests_; }

 private:
  explicit DeepSeekRankProcessManifestPlan(
      std::vector<DeepSeekRankProcessManifest> manifests) noexcept
      : manifests_(std::move(manifests)) {}
  std::vector<DeepSeekRankProcessManifest> manifests_;
};

}  // namespace pih
