#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "pih/model/deepseek_rank_post_mapping_resource_exchange.h"
#include "pih/platform/linux/linux_deepseek_rank_process_driver.h"

namespace pih {

inline constexpr std::string_view
    kLinuxDeepSeekRankPostMappingResourceControllerOperationsAbi =
        "pih_linux_deepseek_rank_post_mapping_resource_controller_operations_v1";
inline constexpr std::string_view
    kLinuxDeepSeekRankPostMappingResourceReporterOperationsAbi =
        "pih_linux_deepseek_rank_post_mapping_resource_reporter_operations_v1";

class LinuxDeepSeekRankPostMappingResourceControllerOperations final
    : public DeepSeekRankPostMappingResourceChannel {
 public:
  static Result<
      LinuxDeepSeekRankPostMappingResourceControllerOperations>
  Create(LinuxDeepSeekRankProcessDriver& driver,
         DeepSeekRankProcessSupervisor& supervisor,
         std::span<const DeepSeekRankProcessHandle> ordered_handles);

  Status send_authority(
      const DeepSeekRankProcessHandle& handle,
      std::span<const std::byte> frame) override;
  Result<std::optional<std::vector<std::byte>>> poll_observation(
      const DeepSeekRankProcessHandle& handle) override;
  Result<std::uint64_t> monotonic_now_ns() override;

 private:
  LinuxDeepSeekRankPostMappingResourceControllerOperations(
      LinuxDeepSeekRankProcessDriver& driver,
      std::vector<DeepSeekRankProcessHandle> handles) noexcept
      : driver_(&driver), handles_(std::move(handles)) {}

  Result<std::uint32_t> rank_for(
      const DeepSeekRankProcessHandle& handle) const;
  Result<std::int32_t> control_fd(std::uint32_t rank) const;

  LinuxDeepSeekRankProcessDriver* driver_ = nullptr;
  std::vector<DeepSeekRankProcessHandle> handles_;
};

class LinuxDeepSeekRankPostMappingResourceReporterOperations final
    : public DeepSeekRankPostMappingResourceReporterOperations {
 public:
  static Result<LinuxDeepSeekRankPostMappingResourceReporterOperations>
  Create(std::int32_t control_fd,
         std::uint64_t expected_controller_process_identity);

  Result<std::optional<std::vector<std::byte>>> receive_authority(
      std::int32_t control_fd) override;
  Result<std::uint64_t> monotonic_now_ns() override;
  Status send_observation(
      std::int32_t control_fd, std::span<const std::byte> frame) override;

 private:
  LinuxDeepSeekRankPostMappingResourceReporterOperations(
      std::int32_t control_fd,
      std::uint64_t controller_process_identity) noexcept
      : control_fd_(control_fd),
        controller_process_identity_(controller_process_identity) {}

  Status validate_bound_control(std::int32_t control_fd) const;

  std::int32_t control_fd_ = -1;
  std::uint64_t controller_process_identity_ = 0;
};

}  // namespace pih
