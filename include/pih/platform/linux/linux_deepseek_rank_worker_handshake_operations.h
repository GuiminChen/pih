#pragma once

#include "pih/model/deepseek_rank_post_exec_resource_exchange.h"
#include "pih/model/deepseek_rank_worker_handshake.h"

namespace pih {

class DeepSeekPhysicalDeviceIdentityProbe {
 public:
  virtual ~DeepSeekPhysicalDeviceIdentityProbe() = default;
  virtual Result<Sha256Digest> current_physical_device_uuid_commitment() = 0;
  virtual Result<std::int32_t> startup_device_ordinal() = 0;
};

class LinuxDeepSeekRankWorkerHandshakeOperations final
    : public DeepSeekRankWorkerHandshakeOperations,
      public DeepSeekRankPostExecResourceReporterOperations {
 public:
  explicit LinuxDeepSeekRankWorkerHandshakeOperations(
      DeepSeekPhysicalDeviceIdentityProbe& device_probe) noexcept
      : device_probe_(&device_probe) {}
  Result<std::optional<std::vector<std::byte>>> receive_challenge(
      std::int32_t control_fd) override;
  Result<DeepSeekRankExecObservation> collect_observation(
      const DeepSeekRankWorkerArguments& arguments) override;
  Status send_ready(std::int32_t control_fd,
                    std::span<const std::byte> frame) override;
  Result<std::optional<std::vector<std::byte>>> receive_authority(
      std::int32_t control_fd) override;
  Result<std::uint64_t> monotonic_now_ns() override;
  Status send_observation(std::int32_t control_fd,
                          std::span<const std::byte> frame) override;

 private:
  DeepSeekPhysicalDeviceIdentityProbe* device_probe_ = nullptr;
};

}  // namespace pih
