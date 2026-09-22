#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "pih/model/deepseek_rank_artifact_metadata_transfer_transaction.h"
#include "pih/platform/linux/linux_deepseek_rank_process_driver.h"

namespace pih {

inline constexpr std::string_view
    kLinuxDeepSeekRankArtifactMetadataTransferControllerOperationsAbi =
        "pih_linux_deepseek_rank_artifact_metadata_transfer_controller_operations_v1";

class LinuxDeepSeekRankArtifactMetadataTransferControllerOperations final
    : public DeepSeekRankArtifactMetadataTransferOperations {
 public:
  static Result<
      LinuxDeepSeekRankArtifactMetadataTransferControllerOperations>
  Create(LinuxDeepSeekRankProcessDriver& driver,
         DeepSeekRankProcessSupervisor& supervisor,
         std::span<const DeepSeekRankProcessHandle> ordered_handles);

  LinuxDeepSeekRankArtifactMetadataTransferControllerOperations(
      const LinuxDeepSeekRankArtifactMetadataTransferControllerOperations&) =
      delete;
  LinuxDeepSeekRankArtifactMetadataTransferControllerOperations& operator=(
      const LinuxDeepSeekRankArtifactMetadataTransferControllerOperations&) =
      delete;
  LinuxDeepSeekRankArtifactMetadataTransferControllerOperations(
      LinuxDeepSeekRankArtifactMetadataTransferControllerOperations&&)
      noexcept = default;
  LinuxDeepSeekRankArtifactMetadataTransferControllerOperations& operator=(
      LinuxDeepSeekRankArtifactMetadataTransferControllerOperations&&)
      noexcept = default;

  Status send_chunk(
      std::uint32_t rank, std::span<const std::byte> frame) override;
  Result<std::optional<std::array<
      std::byte, kDeepSeekRankArtifactMetadataChunkAckFrameBytes>>>
  poll_ack(std::uint32_t rank) override;
  Result<std::uint64_t> monotonic_now_ns() override;
  Status abort_generation(
      std::uint64_t engine_epoch, std::uint64_t worker_generation,
      const Status& cause) override;

 private:
  LinuxDeepSeekRankArtifactMetadataTransferControllerOperations(
      LinuxDeepSeekRankProcessDriver& driver,
      DeepSeekRankProcessSupervisor& supervisor,
      std::vector<DeepSeekRankProcessHandle> handles,
      std::uint64_t engine_epoch,
      std::uint64_t worker_generation) noexcept;

  Result<std::int32_t> control_fd(std::uint32_t rank) const;

  LinuxDeepSeekRankProcessDriver* driver_ = nullptr;
  DeepSeekRankProcessSupervisor* supervisor_ = nullptr;
  std::vector<DeepSeekRankProcessHandle> handles_;
  std::uint64_t engine_epoch_ = 0;
  std::uint64_t worker_generation_ = 0;
};

}  // namespace pih
