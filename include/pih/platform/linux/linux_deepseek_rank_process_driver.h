#pragma once

#include <string>
#include <vector>

#include "pih/model/deepseek_rank_post_exec_resource_exchange.h"
#include "pih/model/deepseek_rank_process_supervisor.h"
#include "pih/model/deepseek_rank_artifact_metadata_blob.h"

namespace pih {

class LinuxDeepSeekRankArtifactTransferControllerOperations;
class LinuxDeepSeekRankArtifactMetadataTransferControllerOperations;
class LinuxDeepSeekRankPostMappingResourceControllerOperations;
class LinuxDeepSeekRankMaterializationControllerOperations;
class LinuxDeepSeekRankServingControllerOperations;

class LinuxDeepSeekRankProcessDriver final
    : public DeepSeekRankProcessDriver,
      public DeepSeekRankPostExecResourceChannel {
 public:
  static Result<LinuxDeepSeekRankProcessDriver> Create(
      std::string worker_executable,
      std::vector<std::string> fixed_arguments,
      std::uint64_t expected_controller_process_identity,
      int inherited_controller_pidfd,
      bool expected_dspark_enabled,
      std::uint64_t maximum_metadata_reassembly_bytes);
  ~LinuxDeepSeekRankProcessDriver() override;
  LinuxDeepSeekRankProcessDriver(const LinuxDeepSeekRankProcessDriver&) = delete;
  LinuxDeepSeekRankProcessDriver& operator=(const LinuxDeepSeekRankProcessDriver&) = delete;
  LinuxDeepSeekRankProcessDriver(LinuxDeepSeekRankProcessDriver&&) noexcept;
  LinuxDeepSeekRankProcessDriver& operator=(LinuxDeepSeekRankProcessDriver&&) noexcept;

  Result<DeepSeekRankProcessHandle> spawn(
      const DeepSeekRankProcessManifest& manifest) override;
  Result<DeepSeekRankProcessObservation> observe(
      const DeepSeekRankProcessHandle& handle) override;
  Status terminate(const DeepSeekRankProcessHandle& handle) override;
  Status send_challenge(const DeepSeekRankProcessHandle& handle,
                        const DeepSeekRankExecChallenge& challenge) override;
  Result<std::optional<DeepSeekRankExecReady>> poll_ready(
      const DeepSeekRankProcessHandle& handle) override;
  Status send_authority(
      const DeepSeekRankProcessHandle& handle,
      std::span<const std::byte> frame) override;
  Result<std::optional<DeepSeekRankPostExecResourceObservation>>
  poll_observation(const DeepSeekRankProcessHandle& handle) override;

 private:
  friend class LinuxDeepSeekRankArtifactTransferControllerOperations;
  friend class
      LinuxDeepSeekRankArtifactMetadataTransferControllerOperations;
  friend class LinuxDeepSeekRankPostMappingResourceControllerOperations;
  friend class LinuxDeepSeekRankMaterializationControllerOperations;
  friend class LinuxDeepSeekRankServingControllerOperations;
  struct OwnedProcess { int pid = -1; int pidfd = -1; int control = -1; };
  LinuxDeepSeekRankProcessDriver(std::string executable,
      std::vector<std::string> arguments, std::uint64_t controller,
      int controller_pidfd, bool expected_dspark_enabled,
      std::uint64_t maximum_metadata_reassembly_bytes) noexcept;
  OwnedProcess* find(const DeepSeekRankProcessHandle& handle) noexcept;
  void reset() noexcept;
  std::string executable_;
  std::vector<std::string> arguments_;
  std::uint64_t controller_ = 0;
  int controller_pidfd_ = -1;
  bool expected_dspark_enabled_ = false;
  std::uint64_t maximum_metadata_reassembly_bytes_ = 0;
  std::vector<OwnedProcess> processes_;
};

}  // namespace pih
