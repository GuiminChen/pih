#include "pih/backend/cuda/atomic_completion_evidence_provider.h"

#include <atomic>

#include <gtest/gtest.h>

namespace pih {
namespace {

class ScriptedLastErrorProbe final : public CompletionLastErrorProbe {
 public:
  Status require_clean_last_error() override { return result; }
  Status result = Status::Ok();
};

TEST(AtomicCompletionEvidenceProviderTest, PublishesJoinedHealthySnapshot) {
  ScriptedLastErrorProbe probe;
  std::atomic<std::uint32_t> device_error{0};
  std::atomic<bool> engine_poisoned{false};
  auto provider = AtomicCompletionEvidenceProvider::Create(
      probe, device_error, engine_poisoned);
  ASSERT_TRUE(provider.ok()) << provider.status().message();

  auto evidence = provider->collect();
  ASSERT_TRUE(evidence.ok()) << evidence.status().message();
  EXPECT_TRUE(evidence->submit_thread_last_error_clean);
  EXPECT_EQ(evidence->device_error_code, 0U);
  EXPECT_FALSE(evidence->engine_poisoned);
}

TEST(AtomicCompletionEvidenceProviderTest, ReadsDeviceAndPoisonAtomically) {
  ScriptedLastErrorProbe probe;
  std::atomic<std::uint32_t> device_error{17};
  std::atomic<bool> engine_poisoned{true};
  auto provider = AtomicCompletionEvidenceProvider::Create(
      probe, device_error, engine_poisoned);
  ASSERT_TRUE(provider.ok());

  auto evidence = provider->collect();
  ASSERT_TRUE(evidence.ok());
  EXPECT_EQ(evidence->device_error_code, 17U);
  EXPECT_TRUE(evidence->engine_poisoned);
}

TEST(AtomicCompletionEvidenceProviderTest, FailsClosedOnDirtySubmitThread) {
  ScriptedLastErrorProbe probe;
  probe.result = Status::Internal("CUDA last error is dirty");
  std::atomic<std::uint32_t> device_error{0};
  std::atomic<bool> engine_poisoned{false};
  auto provider = AtomicCompletionEvidenceProvider::Create(
      probe, device_error, engine_poisoned);
  ASSERT_TRUE(provider.ok());

  auto evidence = provider->collect();
  ASSERT_FALSE(evidence.ok());
  EXPECT_EQ(evidence.status().code(), StatusCode::kInternal);
}

}  // namespace
}  // namespace pih
