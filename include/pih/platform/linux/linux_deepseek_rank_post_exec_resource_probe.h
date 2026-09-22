#pragma once

#include "pih/model/deepseek_rank_post_exec_resource_collector.h"

namespace pih {

class LinuxDeepSeekRankPostExecResourceProbe final
    : public DeepSeekRankPostExecResourceProbe {
 public:
  explicit LinuxDeepSeekRankPostExecResourceProbe(
      DeepSeekRankScmRightsInFlightProbe& scm_rights_probe) noexcept
      : scm_rights_probe_(&scm_rights_probe) {}

  Result<DeepSeekRankPostExecResourceSnapshot> sample() override;

 private:
  DeepSeekRankScmRightsInFlightProbe* scm_rights_probe_ = nullptr;
};

}  // namespace pih
