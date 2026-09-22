#pragma once

#include "pih/model/deepseek_rank_worker_bootstrap.h"

namespace pih {

class LinuxDeepSeekRankWorkerBootstrapRuntime final
    : public DeepSeekRankWorkerBootstrapRuntime {
 public:
  Result<std::uint64_t> monotonic_now_ns() override;
  Status wait_for_control(std::int32_t control_fd,
                          DeepSeekRankWorkerWaitEvent event,
                          std::uint64_t startup_deadline_ns) override;
};

}  // namespace pih
