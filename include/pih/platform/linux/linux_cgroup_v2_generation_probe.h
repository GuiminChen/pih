#pragma once

#include <cstdint>

#include "pih/core/result.h"

namespace pih {

class LinuxCgroupV2GenerationProbe final {
 public:
  static Result<LinuxCgroupV2GenerationProbe> Create(
      std::int32_t generation_cgroup_directory_fd);
  Result<bool> domain_empty() const;

 private:
  explicit LinuxCgroupV2GenerationProbe(std::int32_t directory_fd) noexcept
      : directory_fd_(directory_fd) {}
  std::int32_t directory_fd_ = -1;
};

}  // namespace pih
