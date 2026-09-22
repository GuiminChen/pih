#pragma once

#include "pih/model/deepseek_rank_spawn_resource_collector.h"

namespace pih {

class LinuxDeepSeekRankSpawnAuthorityProbe final
    : public DeepSeekRankSpawnAuthorityProbe {
 public:
  Result<DeepSeekRankSpawnAuthoritySnapshot> read_authority() override;
  Result<DeepSeekRankSpawnUsageSnapshot> sample_usage() override;
};

}  // namespace pih
