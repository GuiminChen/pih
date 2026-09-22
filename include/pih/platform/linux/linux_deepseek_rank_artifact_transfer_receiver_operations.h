#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "pih/model/deepseek_rank_artifact_transfer_receiver.h"
#include "pih/model/deepseek_rank_artifact_metadata_receiver.h"

namespace pih {

inline constexpr std::string_view
    kLinuxDeepSeekRankArtifactTransferReceiverOperationsAbi =
        "pih_linux_deepseek_rank_artifact_transfer_receiver_operations_v1";

class LinuxDeepSeekRankArtifactTransferReceiverOperations final
    : public DeepSeekRankArtifactTransferReceiverOperations,
      public DeepSeekRankArtifactMetadataReceiverOperations {
 public:
  static Result<LinuxDeepSeekRankArtifactTransferReceiverOperations> Create(
      std::int32_t control_fd,
      std::uint64_t expected_controller_process_identity);

  Result<std::optional<std::vector<std::byte>>> receive_manifest(
      std::int32_t control_fd) override;
  Result<std::optional<DeepSeekRankArtifactReceivedBatch>>
  receive_descriptor_batch(std::int32_t control_fd) override;
  Result<std::optional<std::vector<std::byte>>> receive_chunk(
      std::int32_t control_fd) override;
  Result<std::uint64_t> monotonic_now_ns() override;
  Status send_ack(
      std::int32_t control_fd, std::span<const std::byte> frame) override;

 private:
  LinuxDeepSeekRankArtifactTransferReceiverOperations(
      std::int32_t control_fd,
      std::uint64_t controller_process_identity) noexcept
      : control_fd_(control_fd),
        controller_process_identity_(controller_process_identity) {}

  Status validate_bound_control(std::int32_t control_fd) const;

  std::int32_t control_fd_ = -1;
  std::uint64_t controller_process_identity_ = 0;
};

}  // namespace pih
