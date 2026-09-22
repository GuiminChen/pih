#include "pih/platform/linux/linux_deepseek_rank_worker_startup.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

TEST(LinuxDeepSeekRankWorkerStartupTest,
     RejectsMissingDeviceProbeBeforeChangingProcessState) {
  const std::vector<std::string_view> arguments;
  auto startup = LinuxDeepSeekRankWorkerStartup::Create(
      arguments, nullptr);

  EXPECT_FALSE(startup.ok());
  EXPECT_EQ(startup.status().code(), StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace pih
