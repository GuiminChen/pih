#include "pih/platform/linux/linux_deepseek_rank_spawn_authority_probe.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

TEST(LinuxDeepSeekRankSpawnAuthorityProbeTest,
     CollectsUnprivilegedGoverningAuthorities) {
  LinuxDeepSeekRankSpawnAuthorityProbe probe;
  auto authority = probe.read_authority();
  if (!authority.ok() &&
      authority.status().code() == StatusCode::kFailedPrecondition) {
    GTEST_SKIP() << authority.status().message();
  }
  ASSERT_TRUE(authority.ok()) << authority.status().message();
  EXPECT_GT(authority->rlimit_nofile_soft, 0U);
  EXPECT_GE(authority->rlimit_nofile_hard,
            authority->rlimit_nofile_soft);
  EXPECT_GE(authority->fs_nr_open, authority->rlimit_nofile_hard);
  EXPECT_GT(authority->node_file_maximum, 0U);
  EXPECT_GT(authority->vm_max_map_count, 0U);

  auto usage = probe.sample_usage();
  ASSERT_TRUE(usage.ok()) << usage.status().message();
  EXPECT_GT(usage->uid_tasks_current, 0U);
  EXPECT_GT(usage->controller_open_fds, 0U);
  ASSERT_FALSE(usage->cgroup_ancestors.empty());
  EXPECT_GT(usage->cgroup_ancestors.front().current_tasks, 0U);
}

}  // namespace
}  // namespace pih
