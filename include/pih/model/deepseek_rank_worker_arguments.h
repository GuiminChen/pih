#pragma once

#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "pih/model/deepseek_rank_process_supervisor.h"

namespace pih {

struct DeepSeekRankWorkerArguments final {
  DeepSeekRankProcessManifest manifest;
  std::uint64_t controller_process_identity = 0;
  std::int32_t controller_pidfd = -1;
  std::int32_t control_fd = -1;
  bool expected_dspark_enabled = false;
  std::uint64_t maximum_metadata_reassembly_bytes = 0;
  std::vector<std::string> application_arguments;

  static Result<DeepSeekRankWorkerArguments> Parse(
      std::span<const std::string_view> arguments);
};

}  // namespace pih
