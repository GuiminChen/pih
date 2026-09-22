#include "pih/platform/linux/linux_deepseek_rank_post_exec_resource_probe.h"

#include <gtest/gtest.h>

#include <unistd.h>

namespace pih {
namespace {

class ZeroScmRightsProbe final : public DeepSeekRankScmRightsInFlightProbe {
 public:
  Result<std::uint64_t> sample_inflight_fd_count() override {
    ++calls;
    return 0;
  }
  std::uint32_t calls = 0;
};

TEST(LinuxDeepSeekRankPostExecResourceProbeTest,
     SamplesBoundedSelfProcessResources) {
  ZeroScmRightsProbe inflight;
  LinuxDeepSeekRankPostExecResourceProbe probe(inflight);
  auto sample = probe.sample();

  ASSERT_TRUE(sample.ok()) << sample.status().message();
  EXPECT_EQ(sample->process_identity,
            static_cast<std::uint64_t>(::getpid()));
  EXPECT_GT(sample->task_count, 0U);
  EXPECT_GT(sample->open_fd_count, 0U);
  EXPECT_EQ(sample->scm_rights_inflight_fd_count, 0U);
  EXPECT_GT(sample->vma_count, 0U);
  EXPECT_GT(sample->rlimit_nofile_soft, 0U);
  EXPECT_GE(sample->rlimit_nofile_hard, sample->rlimit_nofile_soft);
  EXPECT_GE(sample->fs_nr_open, sample->rlimit_nofile_hard);
  EXPECT_GT(sample->vm_max_map_count, sample->vma_count);
  EXPECT_EQ(inflight.calls, 1U);
}

}  // namespace
}  // namespace pih
