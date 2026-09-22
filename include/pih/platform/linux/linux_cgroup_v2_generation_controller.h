#pragma once

#include <cstdint>

#include <utility>

#include "pih/model/engine_generation_domain_termination.h"
#include "pih/platform/linux/linux_cgroup_v2_generation_probe.h"

namespace pih {

class LinuxCgroupV2GenerationController final
    : public EngineGenerationDomainDriver {
 public:
  static Result<LinuxCgroupV2GenerationController> Create(
      std::int32_t generation_cgroup_directory_fd);
  LinuxCgroupV2GenerationController(
      const LinuxCgroupV2GenerationController&) = delete;
  LinuxCgroupV2GenerationController& operator=(
      const LinuxCgroupV2GenerationController&) = delete;
  LinuxCgroupV2GenerationController(
      LinuxCgroupV2GenerationController&& other) noexcept;
  LinuxCgroupV2GenerationController& operator=(
      LinuxCgroupV2GenerationController&& other) noexcept;
  ~LinuxCgroupV2GenerationController() override;
  Status force_kill_domain() override;
  Result<bool> domain_empty() override;

 private:
  LinuxCgroupV2GenerationController(
      std::int32_t directory_fd,
      LinuxCgroupV2GenerationProbe probe) noexcept
      : directory_fd_(directory_fd), probe_(std::move(probe)) {}
  void close_owned() noexcept;
  std::int32_t directory_fd_ = -1;
  LinuxCgroupV2GenerationProbe probe_;
};

}  // namespace pih
