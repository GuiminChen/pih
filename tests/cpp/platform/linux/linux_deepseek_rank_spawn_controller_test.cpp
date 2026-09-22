#include "pih/platform/linux/linux_deepseek_rank_spawn_controller.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <utility>

#include <unistd.h>

#include "../../model/deepseek_rank_capacity_test_fixture.h"

namespace pih {
namespace {

Sha256Digest controller_digest(std::uint8_t value) {
  Sha256Digest result{};
  result.bytes.fill(static_cast<std::byte>(value));
  return result;
}

TEST(LinuxDeepSeekRankSpawnControllerTest,
     RejectsInvalidDriverBeforeConstructingTheOwnershipGraph) {
  const std::array<DeepSeekRankProcessManifest, 1> manifests{{
      {7, 8, 1, 0, 10, 20, controller_digest(1), 0, 500},
  }};
  const DeepSeekRankSpawnResourcePlan plan{1, 3, 20, 4, 200, 10, 41};
  const std::array<DeepSeekRankPostExecResourcePlan, 1> post_exec{{
      {8, 20, 200, 2, 4, 10, controller_digest(2)},
  }};
  const auto controller_identity = static_cast<std::uint64_t>(::getpid());
  auto capacity = test_fixture::rank_capacity_instance(
      manifests, plan, controller_identity);

  auto controller = LinuxDeepSeekRankSpawnController::Create(
      manifests, plan, post_exec, 700, std::move(capacity),
      "/pih/does-not-exist", {}, -1);

  EXPECT_FALSE(controller.ok());
  EXPECT_EQ(controller.status().code(), StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace pih
