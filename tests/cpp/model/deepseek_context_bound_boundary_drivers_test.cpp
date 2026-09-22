#include "pih/model/deepseek_context_bound_boundary_drivers.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

class BoundaryContextActivator final : public DeepSeekNcclContextActivator {
 public:
  Status activate(std::uintptr_t context) override {
    contexts.push_back(context);
    return status;
  }
  Status status = Status::Ok();
  std::vector<std::uintptr_t> contexts;
};

class BoundaryCompletionDriver final : public CompletionEventDriver,
                                       public CompletionLastErrorProbe {
 public:
  Status record(DriverEventHandle, DriverStreamHandle) override {
    ++record_calls;
    return Status::Ok();
  }
  Result<CudaEventQueryResult> query(DriverEventHandle) override {
    ++query_calls;
    return CudaEventQueryResult::kSuccess;
  }
  Status require_clean_last_error() override {
    ++probe_calls;
    return Status::Ok();
  }
  int record_calls = 0;
  int query_calls = 0;
  int probe_calls = 0;
};

class BoundaryPayload final : public DeepSeekNcclWarmupPayloadOperations {
 public:
  Status prepare(DeepSeekNcclRole, void*, std::uint64_t,
                 DriverStreamHandle, std::uint64_t) override {
    ++prepare_calls;
    return Status::Ok();
  }
  Result<Sha256Digest> digest(const void*, std::uint64_t) override {
    ++digest_calls;
    return Sha256Digest{};
  }
  int prepare_calls = 0;
  int digest_calls = 0;
};

TEST(DeepSeekContextBoundBoundaryDriversTest,
     ActivatesBeforeCompletionAndPayloadOperations) {
  BoundaryContextActivator activator;
  BoundaryCompletionDriver raw_completion;
  BoundaryPayload raw_payload;
  auto completion = DeepSeekContextBoundCompletionDriver::Create(
      77, raw_completion, raw_completion, activator).value();
  auto payload = DeepSeekContextBoundWarmupPayloadOperations::Create(
      77, raw_payload, activator).value();
  std::byte buffer{};
  EXPECT_TRUE(completion.record(1, 2).ok());
  EXPECT_TRUE(completion.query(1).ok());
  EXPECT_TRUE(completion.require_clean_last_error().ok());
  EXPECT_TRUE(payload.prepare(
      DeepSeekNcclRole::kSend, &buffer, 1, 2, 3).ok());
  EXPECT_TRUE(payload.digest(&buffer, 1).ok());
  ASSERT_EQ(activator.contexts.size(), 5U);
  EXPECT_TRUE(std::ranges::all_of(
      activator.contexts, [](std::uintptr_t value) { return value == 77; }));
}

TEST(DeepSeekContextBoundBoundaryDriversTest,
     ActivationFailureSuppressesUnderlyingOperations) {
  BoundaryContextActivator activator;
  activator.status = Status::Internal("activation failed");
  BoundaryCompletionDriver raw_completion;
  BoundaryPayload raw_payload;
  auto completion = DeepSeekContextBoundCompletionDriver::Create(
      77, raw_completion, raw_completion, activator).value();
  auto payload = DeepSeekContextBoundWarmupPayloadOperations::Create(
      77, raw_payload, activator).value();
  std::byte buffer{};
  EXPECT_FALSE(completion.record(1, 2).ok());
  EXPECT_FALSE(payload.digest(&buffer, 1).ok());
  EXPECT_EQ(raw_completion.record_calls, 0);
  EXPECT_EQ(raw_payload.digest_calls, 0);
}

}  // namespace
}  // namespace pih
