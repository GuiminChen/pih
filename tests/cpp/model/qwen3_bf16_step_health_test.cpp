#include "pih/model/qwen3_bf16_step_health.h"

#include <atomic>

#include <gtest/gtest.h>

namespace pih {
namespace {

class Probe final : public QwenBf16SubmitThreadHealthProbe {
 public:
  Status require_clean_last_error() override {
    ++calls;
    return status;
  }
  Status status = Status::Ok();
  int calls = 0;
};

TEST(QwenBf16AtomicStepHealthProviderTest, SamplesProbeAndPoisonAtomically) {
  Probe probe;
  std::atomic<bool> poisoned{false};
  auto provider = QwenBf16AtomicStepHealthProvider::Create(probe, poisoned);
  ASSERT_TRUE(provider.ok());
  auto healthy = provider->collect();
  ASSERT_TRUE(healthy.ok());
  EXPECT_TRUE(healthy->submit_thread_last_error_clean);
  EXPECT_FALSE(healthy->engine_poisoned);

  poisoned.store(true, std::memory_order_release);
  auto failed = provider->collect();
  ASSERT_TRUE(failed.ok());
  EXPECT_TRUE(failed->engine_poisoned);
  EXPECT_EQ(probe.calls, 2);
}

TEST(QwenBf16AtomicStepHealthProviderTest, ProbeFailureCannotPublishHealth) {
  Probe probe;
  probe.status = Status::Internal("sticky CUDA error");
  std::atomic<bool> poisoned{false};
  auto provider = QwenBf16AtomicStepHealthProvider::Create(probe, poisoned);
  ASSERT_TRUE(provider.ok());
  auto health = provider->collect();
  ASSERT_FALSE(health.ok());
  EXPECT_EQ(health.status().code(), StatusCode::kInternal);
}

TEST(QwenBf16TapSnapshotEvidenceProviderTest,
     ProjectsOnlyPreReadbackHealthAxes) {
  Probe probe;
  std::atomic<bool> poisoned{false};
  auto health = QwenBf16AtomicStepHealthProvider::Create(probe, poisoned).value();
  auto evidence = QwenBf16TapSnapshotEvidenceProvider::Create(health).value();

  auto clean = evidence.collect();
  ASSERT_TRUE(clean.ok());
  EXPECT_TRUE(clean->submit_thread_last_error_clean);
  EXPECT_EQ(clean->device_error_code, 0);
  EXPECT_FALSE(clean->engine_poisoned);

  poisoned.store(true, std::memory_order_release);
  auto rejected = evidence.collect();
  ASSERT_TRUE(rejected.ok());
  EXPECT_TRUE(rejected->engine_poisoned);
}

}  // namespace
}  // namespace pih
