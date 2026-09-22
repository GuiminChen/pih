#include "pih/model/deepseek_endpoint_runtime_resources.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

class EndpointPinnedAllocator final : public RegisteredPinnedAllocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    auto* data = _aligned_malloc(static_cast<std::size_t>(bytes),
                                 static_cast<std::size_t>(alignment));
    if (data == nullptr) return Status::ResourceExhausted("endpoint runtime");
    return Allocation{data, bytes, alignment, 5, Device::Cpu()};
  }
  void deallocate(Allocation value) noexcept override { _aligned_free(value.data); }
};

class EndpointOps final : public DeepSeekEndpointSequenceOperations {
 public:
  Status validate_host_error(std::uint32_t* value) override {
    return value == nullptr ? Status::InvalidArgument("null") : Status::Ok();
  }
  Status zero_u32_async(std::uintptr_t, std::uintptr_t) override { return Status::Ok(); }
  Status embedding(DeepSeekEmbeddingLaunch) override { return Status::Ok(); }
  Status hc_head(DeepSeekHcHeadLaunch) override { return Status::Ok(); }
  Status rms_norm(DeepSeekRmsNormLaunch) override { return Status::Ok(); }
  Status lm_head(DeepSeekLmHeadLaunch) override { return Status::Ok(); }
  Status sample(DeepSeekArgmaxLaunch) override { return Status::Ok(); }
  Status copy_token_d2h_async(std::uint32_t*, std::uintptr_t,
                              std::uintptr_t) override { return Status::Ok(); }
  Status copy_error_d2h_async(std::uint32_t*, std::uintptr_t,
                              std::uintptr_t) override { return Status::Ok(); }
};

DeepSeekStagePlan stage(bool embedding, bool head) {
  return {embedding ? 0U : 1U, {embedding ? 0U : 20U, head ? 42U : 21U},
          embedding, head, head};
}

TEST(DeepSeekEndpointRuntimeResourcesTest,
     OwnsExecutorOnlyForEndpointStages) {
  EndpointPinnedAllocator allocator;
  EndpointOps operations;
  auto first = DeepSeekEndpointRuntimeResources::Allocate(
      stage(true, false), operations, allocator);
  ASSERT_TRUE(first.ok());
  EXPECT_NE(first->executor(), nullptr);
  EXPECT_TRUE(first->owns_embedding());
  EXPECT_FALSE(first->owns_lm_head());
  EXPECT_FALSE(first->sampled_token().ok());
  auto last = DeepSeekEndpointRuntimeResources::Allocate(
      stage(false, true), operations, allocator);
  ASSERT_TRUE(last.ok());
  EXPECT_TRUE(last->owns_lm_head());
  EXPECT_EQ(last->sampled_token().value(), 0U);
  ASSERT_TRUE(last->sampling_result().ok());
  EXPECT_EQ(last->sampling_result()->token_id, 0U);
  EXPECT_FLOAT_EQ(last->sampling_result()->selected_logprob, 0.0F);
  EXPECT_EQ(last->sampling_result()->rng_word, 0U);
  auto top = last->sampling_result(2);
  ASSERT_TRUE(top.ok());
  EXPECT_EQ(top->top_token_ids, std::vector<std::uint32_t>({0, 0}));
  EXPECT_EQ(top->top_logprobs, std::vector<float>({0.0F, 0.0F}));
  auto middle = DeepSeekEndpointRuntimeResources::Allocate(
      stage(false, false), operations, allocator);
  ASSERT_TRUE(middle.ok());
  EXPECT_EQ(middle->executor(), nullptr);
}

TEST(DeepSeekEndpointRuntimeResourcesTest, RejectsMissingEndpointOperations) {
  EndpointPinnedAllocator allocator;
  EXPECT_FALSE(DeepSeekEndpointRuntimeResources::Allocate(
      stage(false, true), nullptr, allocator).ok());
}

}  // namespace
}  // namespace pih
