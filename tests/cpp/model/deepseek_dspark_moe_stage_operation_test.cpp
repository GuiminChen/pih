#include "pih/model/deepseek_dspark_moe_stage_operation.h"

#include <gtest/gtest.h>

#include <array>
#include <string>
#include <vector>

#include "deepseek_dspark_weight_test_fixture.h"

namespace pih {
namespace {

Result<TensorView> resident_tensor(std::string_view name) {
  const bool scale = name.ends_with(".scale");
  const bool w2 = name.find(".w2.") != std::string_view::npos;
  const std::array<std::int64_t, 2> shape = scale
      ? std::array<std::int64_t, 2>{w2 ? 4096 : 2048, w2 ? 64 : 128}
      : std::array<std::int64_t, 2>{w2 ? 4096 : 2048,
                                    w2 ? 1024 : 2048};
  return TensorView::Create(
      reinterpret_cast<void*>(0x10000U +
          (std::hash<std::string_view>{}(name) & 0x0fffffffU)),
      scale ? DType::kFloat8E8M0 : DType::kInt8, shape, {},
      Device::Create(DeviceType::kCuda, 0).value(), 29);
}

class FixedOperations final : public DeepSeekFixedStateBankOperations {
 public:
  Status copy_d2d_async(std::uintptr_t, std::uintptr_t, std::uint64_t,
                        std::uintptr_t) override { return Status::Ok(); }
  Status record_event(std::uintptr_t, std::uintptr_t) override {
    return Status::Ok();
  }
  Result<DeepSeekExpertAsyncStatus> query_event(std::uintptr_t) override {
    return DeepSeekExpertAsyncStatus::kSuccess;
  }
};

class Operations final : public DeepSeekDsparkMoeStageOperations {
 public:
  Status validate_host_error(std::uint32_t*) override { return Status::Ok(); }
  Status zero_u32_async(std::uintptr_t, std::uintptr_t) override {
    calls.push_back("zero_error"); return Status::Ok();
  }
  Status zero_bytes_async(std::uintptr_t, std::uint64_t bytes,
                          std::uintptr_t) override {
    calls.push_back("zero_acc"); zero_bytes = bytes; return Status::Ok();
  }
  Status mhc_pre(DeepSeekMhcPreLaunch launch) override {
    calls.push_back("mhc_pre"); pre = launch; return Status::Ok();
  }
  Status router_gemm(DeepSeekRouterBf16GemmLaunch launch) override {
    calls.push_back("router"); router = launch; return Status::Ok();
  }
  Status copy_d2h_async(void*, std::uintptr_t, std::uint64_t,
                        std::uintptr_t) override {
    calls.push_back("d2h"); return Status::Ok();
  }
  Status record_event(std::uintptr_t, std::uintptr_t) override {
    calls.push_back("event"); return Status::Ok();
  }
  Result<DeepSeekExpertAsyncStatus> query_event(std::uintptr_t) override {
    return event_status;
  }
  Status quant(DeepSeekFp8ActivationQuantLaunch) override {
    calls.push_back("quant"); return Status::Ok();
  }
  Status fp4_gemm(DeepSeekFp4GemmLaunch) override {
    calls.push_back("fp4"); return Status::Ok();
  }
  Status shared_swiglu(DeepSeekSharedExpertSwiGluLaunch) override {
    calls.push_back("shared_swiglu"); return Status::Ok();
  }
  Status finalize(DeepSeekExpertFinalizeLaunch launch) override {
    calls.push_back("finalize"); finalize_launch = launch;
    return Status::Ok();
  }
  Status mhc_post(DeepSeekMhcPostLaunch launch) override {
    calls.push_back("mhc_post"); post = launch; return Status::Ok();
  }
  Status synchronize_stream(std::uintptr_t) override {
    calls.push_back("sync"); return Status::Ok();
  }
  std::vector<std::string> calls;
  DeepSeekExpertAsyncStatus event_status =
      DeepSeekExpertAsyncStatus::kSuccess;
  std::uint64_t zero_bytes = 0;
  DeepSeekMhcPreLaunch pre;
  DeepSeekRouterBf16GemmLaunch router;
  DeepSeekExpertFinalizeLaunch finalize_launch;
  DeepSeekMhcPostLaunch post;
};

class Kernel final : public DeepSeekDsparkExpertKernelDriver {
 public:
  Status launch_resident(
      DeepSeekDsparkStageId stage, std::uint16_t expert,
      std::uint64_t generation,
      const DeepSeekExpertBundleDeviceView&,
      const DeepSeekExpertRoute*, std::uint32_t) override {
    stages.push_back(stage); experts.push_back(expert);
    generations.push_back(generation); return Status::Ok();
  }
  Result<DeepSeekExpertAsyncStatus> poll() override {
    return DeepSeekExpertAsyncStatus::kSuccess;
  }
  std::vector<DeepSeekDsparkStageId> stages;
  std::vector<std::uint16_t> experts;
  std::vector<std::uint64_t> generations;
};

struct Fixture final {
  Fixture()
      : weights(test::deepseek_dspark_test_weights()),
        experts(DeepSeekDsparkResidentExpertBindings::Resolve(
                    test::deepseek_dspark_test_stage(), resident_tensor)
                    .value()) {
    EXPECT_TRUE(transaction.begin(11).ok());
    scores.fill(0.0F);
    bias.fill(0.0F);
    for (std::size_t index = 0; index < 6; ++index) bias[index] = 10.0F;
    auto layout_value = DeepSeekExpertComputeArenaLayout::Create(5).value();
    arena = layout_value.bind(0x30000000U,
                              layout_value.required_bytes()).value();
    const auto stage = DeepSeekDsparkStageId::kMtp1;
    resources.stage = stage;
    resources.weights = &weights.common(stage);
    resources.resident_experts = &experts;
    resources.state_layout = &state_layout;
    resources.transaction = &transaction;
    resources.prepare_epoch = transaction.prepare_epoch();
    resources.recent_state = state_layout.ResolveDspark(
        stage, transaction.tentative_fixed_state().value()).value();
    resources.weight_generation = weights.generation();
    resources.mhc_workspace = {
        0x401, 0x402, 0x403, 0x404, 0x405, 0x406};
    resources.attention_output_hc_bf16 = 0x501;
    resources.stage_output_hc_bf16 = 0x502;
    resources.error_flag_u32 = 0x503;
    resources.stream = 11;
    resources.completion_event = 12;
    resources.router_scores_f32 = 0x504;
    resources.router_host_scores = scores;
    resources.router_host_bias = bias;
    resources.expert_arena = arena;
    resources.expert_accumulator_f32 = 0x505;
    resources.expert_kernel = &kernel;
  }

  static constexpr std::uint64_t kStateBytes =
      kDeepSeekDsparkStageCount * kDeepSeekDsparkRecentStateBytes;
  FixedOperations fixed_operations;
  DeepSeekFixedStateLayout state_layout =
      DeepSeekFixedStateLayout::Build({}, true).value();
  DeepSeekFixedStateBanks banks = DeepSeekFixedStateBanks::Create(
      {0x100000, kStateBytes}, {0x200000, kStateBytes}, 121,
      fixed_operations).value();
  DeepSeekRatio4PagePool ratio4 = DeepSeekRatio4PagePool::Create(1).value();
  DeepSeekRatio128PagePool ratio128 =
      DeepSeekRatio128PagePool::Create(1).value();
  DeepSeekAttentionSequenceTransaction transaction =
      DeepSeekAttentionSequenceTransaction::Create(
          1, 0, 0, banks, ratio4, ratio128).value();
  DeepSeekDsparkWeightBindings weights;
  DeepSeekDsparkResidentExpertBindings experts;
  std::array<float, 5 * 256> scores;
  std::array<float, 256> bias;
  DeepSeekExpertComputeArena arena;
  Operations operations;
  Kernel kernel;
  DeepSeekDsparkMtpStageResources resources;
  std::uint32_t host_error = 0;
};

TEST(DeepSeekDsparkMoeStageOperationTest,
     ExecutesRouterRoutedSharedAndMhcPostInOrder) {
  Fixture fixture;
  auto operation = DeepSeekDsparkMoeStageOperation::Create(
      DeepSeekDsparkStageId::kMtp1, fixture.operations,
      &fixture.host_error).value();
  ASSERT_TRUE(operation.launch(
      {9, 17, DeepSeekPlanPhase::kDecode, 1, 1},
      fixture.resources).ok());
  EXPECT_EQ(fixture.operations.calls,
            (std::vector<std::string>{
                "zero_error", "zero_acc", "mhc_pre", "router",
                "d2h", "d2h", "d2h", "event"}));
  EXPECT_EQ(fixture.operations.zero_bytes, 5U * 4096U * sizeof(float));

  DeepSeekStageComputeStatus status = DeepSeekStageComputeStatus::kInProgress;
  for (std::uint32_t poll = 0; poll < 16 &&
       status == DeepSeekStageComputeStatus::kInProgress; ++poll) {
    status = operation.poll().value();
  }
  EXPECT_EQ(status, DeepSeekStageComputeStatus::kSuccess);
  EXPECT_EQ(fixture.kernel.experts,
            (std::vector<std::uint16_t>{0, 1, 2, 3, 4, 5}));
  EXPECT_EQ(fixture.kernel.stages,
            std::vector<DeepSeekDsparkStageId>(
                6, DeepSeekDsparkStageId::kMtp1));
  EXPECT_EQ(fixture.operations.post.output_bf16,
            fixture.resources.stage_output_hc_bf16);
  EXPECT_EQ(fixture.operations.finalize_launch.accumulator_f32,
            fixture.resources.expert_accumulator_f32);
  EXPECT_EQ(fixture.operations.calls,
            (std::vector<std::string>{
                "zero_error", "zero_acc", "mhc_pre", "router",
                "d2h", "d2h", "d2h", "event", "quant", "fp4",
                "fp4", "shared_swiglu", "quant", "fp4", "finalize",
                "mhc_post", "d2h", "event"}));
}

TEST(DeepSeekDsparkMoeStageOperationTest,
     WaitsForRouterEventAndRejectsMalformedHostGeometry) {
  Fixture fixture;
  fixture.operations.event_status = DeepSeekExpertAsyncStatus::kInProgress;
  auto operation = DeepSeekDsparkMoeStageOperation::Create(
      DeepSeekDsparkStageId::kMtp1, fixture.operations,
      &fixture.host_error).value();
  ASSERT_TRUE(operation.launch(
      {9, 17, DeepSeekPlanPhase::kDecode, 1, 1}, fixture.resources).ok());
  EXPECT_EQ(operation.poll().value(), DeepSeekStageComputeStatus::kInProgress);
  EXPECT_TRUE(fixture.kernel.experts.empty());

  Fixture malformed;
  malformed.resources.router_host_bias = {};
  auto rejected = DeepSeekDsparkMoeStageOperation::Create(
      DeepSeekDsparkStageId::kMtp1, malformed.operations,
      &malformed.host_error).value();
  EXPECT_FALSE(rejected.launch(
      {9, 17, DeepSeekPlanPhase::kDecode, 1, 1},
      malformed.resources).ok());
}

TEST(DeepSeekDsparkMoeStageOperationTest,
     CancellationSynchronizesAndPoisonsTheOperation) {
  Fixture fixture;
  fixture.operations.event_status = DeepSeekExpertAsyncStatus::kInProgress;
  auto operation = DeepSeekDsparkMoeStageOperation::Create(
      DeepSeekDsparkStageId::kMtp1, fixture.operations,
      &fixture.host_error).value();
  ASSERT_TRUE(operation.launch(
      {9, 17, DeepSeekPlanPhase::kDecode, 1, 1}, fixture.resources).ok());
  EXPECT_TRUE(operation.cancel().ok());
  EXPECT_EQ(fixture.operations.calls.back(), "sync");
  EXPECT_FALSE(operation.poll().ok());
  EXPECT_FALSE(operation.launch(
      {9, 18, DeepSeekPlanPhase::kDecode, 1, 1}, fixture.resources).ok());
}

}  // namespace
}  // namespace pih
