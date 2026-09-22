#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

#include "pih/backend/cuda/nvidia_deepseek_rank_runtime_view.h"
#include "pih/model/deepseek_capability_artifact_catalog.h"
#include "pih/model/deepseek_engine.h"
#include "pih/model/deepseek_execution_topology.h"
#include "pih/model/runtime_profile_payload.h"

namespace pih {

class NvidiaDeepSeekBootstrapConfig;
class DeepSeekOptimizationPolicy;

class NvidiaDeepSeekRankRuntimeFactory {
 public:
  virtual ~NvidiaDeepSeekRankRuntimeFactory() = default;
  virtual Result<std::unique_ptr<NvidiaDeepSeekRankRuntimeView>> Create(
      std::int32_t device_ordinal, std::uint32_t rank,
      std::uint64_t worker_generation,
      const DeepSeekOptimizationPolicy& optimization_policy) = 0;
};

class NvidiaDeepSeekBootstrapConfig final {
 public:
  static Result<NvidiaDeepSeekBootstrapConfig> Create(
      std::uint64_t epoch, std::uint32_t world_size,
      std::vector<std::int32_t> device_ordinals,
      DeepSeekRoutedExpertResidency expert_residency,
      std::uint32_t expert_slot_count,
      std::uint32_t staging_extent_count,
      std::uint32_t artifact_poll_interval_ms,
      std::uint32_t attention_reserved_tokens_per_sequence = 4096,
      DeepSeekExecutionTopology execution_topology =
          DeepSeekExecutionTopology::kInProcessRankSetDevelopment,
      std::uint64_t host_spill_pinned_bytes = 0,
      std::uint64_t host_spill_device_bytes = 0,
      std::uint32_t transfer_reservation_window = 0);

  [[nodiscard]] std::uint64_t epoch() const noexcept { return epoch_; }
  [[nodiscard]] std::uint32_t world_size() const noexcept { return world_size_; }
  [[nodiscard]] const std::vector<std::int32_t>& device_ordinals() const noexcept {
    return device_ordinals_;
  }
  [[nodiscard]] DeepSeekRoutedExpertResidency expert_residency() const noexcept {
    return expert_residency_;
  }
  [[nodiscard]] std::uint32_t expert_slot_count() const noexcept {
    return expert_slot_count_;
  }
  [[nodiscard]] std::uint32_t staging_extent_count() const noexcept {
    return staging_extent_count_;
  }
  [[nodiscard]] std::uint32_t artifact_poll_interval_ms() const noexcept {
    return artifact_poll_interval_ms_;
  }
  [[nodiscard]] std::uint32_t attention_reserved_tokens_per_sequence()
      const noexcept { return attention_reserved_tokens_per_sequence_; }
  [[nodiscard]] std::uint64_t host_spill_pinned_bytes() const noexcept {
    return host_spill_pinned_bytes_;
  }
  [[nodiscard]] std::uint64_t host_spill_device_bytes() const noexcept {
    return host_spill_device_bytes_;
  }
  [[nodiscard]] std::uint32_t transfer_reservation_window() const noexcept {
    return transfer_reservation_window_;
  }
  [[nodiscard]] DeepSeekExecutionTopology execution_topology() const noexcept {
    return execution_topology_;
  }
  [[nodiscard]] bool in_process_compute_supported() const noexcept {
    return execution_topology_ ==
               DeepSeekExecutionTopology::kInProcessRankSetDevelopment &&
           world_size_ >= 1 && world_size_ <= 4 &&
           (expert_residency_ == DeepSeekRoutedExpertResidency::kHostSpill ||
            expert_residency_ ==
                DeepSeekRoutedExpertResidency::kFullResident);
  }
  [[nodiscard]] bool production_compute_supported() const noexcept {
    return false;
  }

 private:
  std::uint64_t epoch_ = 0;
  std::uint32_t world_size_ = 0;
  std::vector<std::int32_t> device_ordinals_;
  DeepSeekRoutedExpertResidency expert_residency_ =
      DeepSeekRoutedExpertResidency::kHostSpill;
  std::uint32_t expert_slot_count_ = 0;
  std::uint32_t staging_extent_count_ = 0;
  std::uint32_t artifact_poll_interval_ms_ = 0;
  std::uint32_t attention_reserved_tokens_per_sequence_ = 0;
  std::uint64_t host_spill_pinned_bytes_ = 0;
  std::uint64_t host_spill_device_bytes_ = 0;
  std::uint32_t transfer_reservation_window_ = 0;
  DeepSeekExecutionTopology execution_topology_ =
      DeepSeekExecutionTopology::kInProcessRankSetDevelopment;
};

class NvidiaDeepSeekEngineBootstrap final {
 public:
  static Result<std::unique_ptr<DeepSeekEngine>>
  BuildDevelopmentSm89Pp1FromArtifactCapabilityWithRuntimeFactory(
      const pih_verified_artifact_api_v1& artifact_api,
      const std::filesystem::path& trusted_generation_root,
      const Sha256Digest& expected_artifact_root,
      std::uint64_t maximum_shard_bytes, DeepSeekPipelineCapacity capacity,
      const NvidiaDeepSeekBootstrapConfig& config,
      NvidiaDeepSeekRankRuntimeFactory& runtime_factory);
  static Result<std::unique_ptr<DeepSeekEngine>>
  BuildDevelopmentSm89Pp1FromArtifactCatalogWithRuntimeFactory(
      DeepSeekCapabilityArtifactCatalog catalog,
      DeepSeekPipelineCapacity capacity,
      const NvidiaDeepSeekBootstrapConfig& config,
      NvidiaDeepSeekRankRuntimeFactory& runtime_factory);

};

}  // namespace pih
