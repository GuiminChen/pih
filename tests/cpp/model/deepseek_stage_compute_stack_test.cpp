#include "pih/model/deepseek_stage_compute_stack.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

class DecodeProvider final : public DeepSeekDecodeAttentionWorkProvider {
 public:
  Result<const DeepSeekDecodeAttentionWork*> resolve(
      std::uint32_t, const DeepSeekPipelinePlanDescriptor&) override {
    return Status::Unavailable("unused decode work");
  }
};
class ChunkProvider final : public DeepSeekChunkAttentionWorkProvider {
 public:
  Result<const DeepSeekChunkAttentionWork*> resolve(
      std::uint32_t, const DeepSeekPipelinePlanDescriptor&) override {
    return Status::Unavailable("unused chunk work");
  }
};
class DenseProvider final : public DeepSeekDenseAttentionStageWorkProvider {
 public:
  Result<std::span<const DeepSeekDenseAttentionStageSequenceWork>> resolve(
      const DeepSeekStageOperatorCommand&,
      const DeepSeekPipelinePlanDescriptor&) override {
    return Status::Unavailable("unused dense work");
  }
};
class RouterProvider final : public DeepSeekRouterStageWorkProvider {
 public:
  explicit RouterProvider(DeepSeekExpertPlanProvider& provider)
      : provider_(&provider) {}
  DeepSeekExpertPlanProvider* plan_provider() noexcept override {
    return provider_;
  }
  Result<DeepSeekHashRouterStageWork> resolve_hash(
      const DeepSeekStageOperatorCommand&,
      const DeepSeekPipelinePlanDescriptor&) override {
    return Status::Unavailable("unused hash work");
  }
  Result<DeepSeekLearnedRouterStageWork> resolve_learned(
      const DeepSeekStageOperatorCommand&,
      const DeepSeekPipelinePlanDescriptor&) override {
    return Status::Unavailable("unused learned work");
  }
 private:
  DeepSeekExpertPlanProvider* provider_;
};
class ExpertProvider final : public DeepSeekExpertPlanProvider {
 public:
  Result<const DeepSeekExpertSubwavePlan*> resolve(
      std::uint32_t, const DeepSeekPipelinePlanDescriptor&) override {
    return Status::Unavailable("unused expert work");
  }
};
class MhcProvider final : public DeepSeekMhcStageWorkProvider {
 public:
  Result<std::span<const DeepSeekMhcStageSequenceWork>> resolve(
      const DeepSeekStageOperatorCommand&,
      const DeepSeekPipelinePlanDescriptor&) override {
    return Status::Unavailable("unused mhc work");
  }
};
class EndpointProvider final : public DeepSeekEndpointStageWorkProvider {
 public:
  Result<std::span<const DeepSeekEndpointStageSequenceWork>> resolve(
      const DeepSeekStageOperatorCommand&,
      const DeepSeekPipelinePlanDescriptor&) override {
    return Status::Unavailable("unused endpoint work");
  }
};
class DsparkProvider final : public DeepSeekDsparkStageWorkProvider {
 public:
  Result<const DeepSeekDsparkStageWork*> resolve(
      const DeepSeekPipelinePlanDescriptor&) override {
    return Status::Unavailable("unused dspark work");
  }
};
class DsparkMtpBackend final : public DeepSeekDsparkMtpOperatorBackend {
 public:
  Status launch(const DeepSeekDsparkMtpOperatorCommand&,
                const DeepSeekPipelinePlanDescriptor&) override {
    return Status::Unavailable("unused dspark mtp work");
  }
  Result<DeepSeekStageComputeStatus> poll() override {
    return Status::Unavailable("unused dspark mtp work");
  }
  Status cancel() override {
    return Status::Unavailable("unused dspark mtp work");
  }
};

class Transfer final : public DeepSeekExpertTransferDriver {
 public:
  Status start(DeepSeekExpertIdentity, std::uint32_t, std::uint64_t,
               std::uint64_t) override { return Status::Ok(); }
  Result<DeepSeekExpertAsyncStatus> poll(
      DeepSeekExpertIdentity, std::uint64_t) override {
    return DeepSeekExpertAsyncStatus::kSuccess;
  }
};

class Kernel final : public DeepSeekExpertKernelDriver {
 public:
  Status launch(const DeepSeekExpertLease&, const DeepSeekExpertRoute*,
                std::uint32_t) override { return Status::Ok(); }
  Result<DeepSeekExpertAsyncStatus> poll() override {
    return DeepSeekExpertAsyncStatus::kSuccess;
  }
};

Result<TensorView> resident_tensor(std::string_view name) {
  const bool scale = name.ends_with(".scale");
  const bool w2 = name.find(".w2.") != std::string_view::npos;
  const std::array<std::int64_t, 2> shape = scale
      ? std::array<std::int64_t, 2>{w2 ? 4096 : 2048, w2 ? 64 : 128}
      : std::array<std::int64_t, 2>{w2 ? 4096 : 2048, w2 ? 1024 : 2048};
  return TensorView::Create(
      reinterpret_cast<void*>(0x10000U +
          (std::hash<std::string_view>{}(name) & 0x0fffffffU)),
      scale ? DType::kFloat8E8M0 : DType::kInt8, shape, {},
      Device::Create(DeviceType::kCuda, 0).value(), 17);
}

TEST(DeepSeekStageComputeStackTest, RejectsIncompleteProductionDependencies) {
  auto plan = DeepSeekPipelinePlan::Create(1, true);
  ASSERT_TRUE(plan.ok());
  EXPECT_EQ(DeepSeekStageComputeStack::Create(plan->rank(0), {}).status().code(),
            StatusCode::kInvalidArgument);
}

TEST(DeepSeekStageComputeStackTest,
     OwnsStableCanonicalChainAcrossMoveAndDrain) {
  DecodeProvider decode;
  ChunkProvider chunk;
  DenseProvider dense;
  ExpertProvider expert;
  RouterProvider router(expert);
  MhcProvider mhc;
  EndpointProvider endpoint;
  DsparkProvider dspark;
  DsparkMtpBackend dspark_mtp;
  Transfer transfer;
  Kernel kernel;
  auto pager = DeepSeekExpertPager::Create({0, 42}, 2, 2);
  ASSERT_TRUE(pager.ok());
  DeepSeekStageComputeStackDependencies dependencies;
  dependencies.decode_attention = &decode;
  dependencies.chunk_attention = &chunk;
  dependencies.dense_attention = &dense;
  dependencies.router = &router;
  dependencies.expert_plan = &expert;
  dependencies.expert_pager = &*pager;
  dependencies.expert_transfer = &transfer;
  dependencies.expert_kernel = &kernel;
  dependencies.mhc = &mhc;
  dependencies.endpoint = &endpoint;
  dependencies.dspark = &dspark;
  dependencies.dspark_mtp = &dspark_mtp;
  ExpertProvider foreign;
  dependencies.expert_plan = &foreign;
  auto plan = DeepSeekPipelinePlan::Create(1, true);
  ASSERT_TRUE(plan.ok());
  EXPECT_EQ(DeepSeekStageComputeStack::Create(
                plan->rank(0), dependencies).status().code(),
            StatusCode::kInvalidArgument);
  dependencies.expert_plan = &expert;
  auto pipeline3 = DeepSeekPipelinePlan::Create(3, false);
  ASSERT_TRUE(pipeline3.ok());
  dependencies.endpoint = nullptr;
  dependencies.dspark = nullptr;
  dependencies.dspark_mtp = nullptr;
  auto middle = DeepSeekStageComputeStack::Create(
      pipeline3->rank(1), dependencies);
  ASSERT_TRUE(middle.ok()) << middle.status().message();
  // Missing shared-expert assembly must fail before any inference work. Drain
  // remains available so a partially initialized runtime can be cleaned up.
  EXPECT_FALSE(middle->launch(
      {5, 1, DeepSeekPlanPhase::kDecode, 1, 1}, pipeline3->rank(1)).ok());
  ASSERT_TRUE(middle->launch(
      {5, 2, DeepSeekPlanPhase::kDrain, 0, 1}, pipeline3->rank(1)).ok());
  ASSERT_EQ(middle->poll().value(), DeepSeekStageComputeStatus::kSuccess);
  dependencies.endpoint = &endpoint;
  dependencies.dspark = &dspark;
  dependencies.dspark_mtp = &dspark_mtp;
  auto created = DeepSeekStageComputeStack::Create(
      plan->rank(0), dependencies);
  ASSERT_TRUE(created.ok()) << created.status().message();
  DeepSeekStageComputeStack moved(std::move(*created));

  ASSERT_TRUE(moved.launch(
      {5, 1, DeepSeekPlanPhase::kDrain, 0, 1}, plan->rank(0)).ok());
  auto result = moved.poll();
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(*result, DeepSeekStageComputeStatus::kSuccess);
}

TEST(DeepSeekStageComputeStackTest,
     ResidentDependenciesDoNotRequirePagerOrTransfer) {
  auto pipeline = DeepSeekPipelinePlan::Create(3, false).value();
  const auto stage = pipeline.rank(1);
  auto resident = DeepSeekResidentExpertBindings::Resolve(
      stage, resident_tensor).value();
  DecodeProvider decode; ChunkProvider chunk; DenseProvider dense;
  ExpertProvider expert; RouterProvider router(expert); MhcProvider mhc;
  Kernel kernel;
  DeepSeekStageComputeStackDependencies dependencies;
  dependencies.decode_attention = &decode;
  dependencies.chunk_attention = &chunk;
  dependencies.dense_attention = &dense;
  dependencies.router = &router;
  dependencies.expert_plan = &expert;
  dependencies.resident_experts = &resident;
  dependencies.expert_kernel = &kernel;
  dependencies.mhc = &mhc;
  auto stack = DeepSeekStageComputeStack::Create(stage, dependencies);
  ASSERT_TRUE(stack.ok()) << stack.status().message();
  ASSERT_TRUE(stack->launch(
      {7, 1, DeepSeekPlanPhase::kDrain, 0, 1}, stage).ok());
  EXPECT_EQ(stack->poll().value(), DeepSeekStageComputeStatus::kSuccess);
}

}  // namespace
}  // namespace pih
