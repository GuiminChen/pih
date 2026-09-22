#include "pih/model/deepseek_shared_expert_driver.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace pih { namespace {

Result<TensorView> shared_tensor(std::string_view name) {
  const std::string prefix = "layers.2.ffn.shared_experts.";
  if (!name.starts_with(prefix)) return Status::InvalidArgument("unknown layer");
  const auto suffix = name.substr(prefix.size());
  const bool down = suffix.starts_with("w2.");
  if (!down && !suffix.starts_with("w1.") && !suffix.starts_with("w3.")) {
    return Status::InvalidArgument("unknown matrix");
  }
  const bool scale = suffix.ends_with(".scale");
  if (!scale && !suffix.ends_with(".weight")) return Status::InvalidArgument("unknown member");
  const std::int64_t divisor = scale ? 128 : 1;
  const std::vector<std::int64_t> shape{
      (down ? 4096 : 2048) / divisor, (down ? 2048 : 4096) / divisor};
  const auto ordinal = (down ? 4U : suffix.starts_with("w3.") ? 2U : 0U) + scale;
  return TensorView::Create(reinterpret_cast<void*>(0x1000000ULL * (ordinal + 1)),
      scale ? DType::kFloat8E8M0 : DType::kFloat8E4M3, shape, {},
      Device::Create(DeviceType::kCuda, 1).value(), 73);
}

TEST(DeepSeekSharedExpertWeightsTest, ResolvesAllSixRawFp8Tensors) {
  auto weights = DeepSeekSharedExpertWeights::Resolve(2, shared_tensor);
  ASSERT_TRUE(weights.ok()) << weights.status().message();
  EXPECT_EQ(weights->generation, 73U);
  EXPECT_EQ(weights->device_ordinal, 1);
  EXPECT_NE(weights->w1_e4m3, weights->w1_scale_bits);
  EXPECT_NE(weights->w2_e4m3, weights->w3_e4m3);
  EXPECT_FALSE(DeepSeekSharedExpertWeights::Resolve(43, shared_tensor).ok());
  EXPECT_FALSE(DeepSeekSharedExpertWeights::Resolve(
      2, DeepSeekSharedExpertWeights::Resolver{}).ok());
}

TEST(DeepSeekSharedExpertWeightsTest, RejectsMissingScaleAndForeignIdentity) {
  auto missing = [](std::string_view name) -> Result<TensorView> {
    if (name.ends_with("w2.scale")) return Status::FailedPrecondition("missing scale");
    return shared_tensor(name);
  };
  EXPECT_FALSE(DeepSeekSharedExpertWeights::Resolve(2, missing).ok());
  for (const bool change_device : {false, true}) {
    auto foreign = [change_device](std::string_view name) -> Result<TensorView> {
      auto view = shared_tensor(name);
      if (!view.ok() || !name.ends_with("w2.scale")) return view;
      return TensorView::Create(view->data(), view->dtype(),
          std::vector<std::int64_t>{32, 16}, {},
          Device::Create(DeviceType::kCuda, change_device ? 0 : 1).value(),
          change_device ? 73 : 74);
    };
    EXPECT_FALSE(DeepSeekSharedExpertWeights::Resolve(2, foreign).ok());
  }
}

TEST(DeepSeekSharedExpertWeightsTest, RejectsPackedFp4AndScaleShapeDrift) {
  for (const bool change_dtype : {false, true}) {
    auto drift = [change_dtype](std::string_view name) -> Result<TensorView> {
      auto view = shared_tensor(name);
      if (!view.ok()) return view;
      if (change_dtype && name.ends_with("w1.weight")) {
        return TensorView::Create(view->data(), DType::kInt8,
            std::vector<std::int64_t>{2048, 2048}, {}, view->device(), 73);
      }
      if (!change_dtype && name.ends_with("w1.scale")) {
        return TensorView::Create(view->data(), view->dtype(),
            std::vector<std::int64_t>{2048, 128}, {}, view->device(), 73);
      }
      return view;
    };
    EXPECT_FALSE(DeepSeekSharedExpertWeights::Resolve(2, drift).ok());
  }
}

class Operations final : public DeepSeekSharedExpertOperations {
 public:
  Status validate_host_error(std::uint32_t*) override { return Status::Ok(); }
  Status prepare_moe(std::uintptr_t, std::uintptr_t, std::uintptr_t,
      std::uint32_t, std::uintptr_t) override { return Status::Ok(); }
  Status copy_error_d2h_async(std::uint32_t* host, std::uintptr_t,
      std::uintptr_t) override {
    copied_error = host; calls.push_back("copy"); return next();
  }
  Status record_event(std::uintptr_t, std::uintptr_t) override {
    calls.push_back("event"); return next();
  }
  Result<DeepSeekExpertAsyncStatus> query_event(std::uintptr_t) override {
    if (completion == DeepSeekExpertAsyncStatus::kSuccess && copied_error) {
      *copied_error = device_error;
    }
    return completion;
  }
  std::uint32_t host_error = 0;
  std::uint32_t device_error = 0;
  std::uint32_t* copied_error = nullptr;
  DeepSeekExpertAsyncStatus completion = DeepSeekExpertAsyncStatus::kSuccess;
  Status zero_u32_async(std::uintptr_t, std::uintptr_t) override {
    calls.push_back("zero"); return next();
  }
  Status quantize(DeepSeekFp8ActivationQuantLaunch value) override {
    calls.push_back("quant"); quant.push_back(value); return next();
  }
  Status gemm(DeepSeekFp8GemmLaunch value) override {
    calls.push_back("gemm"); gemms.push_back(value); return next();
  }
  Status swiglu(DeepSeekSharedExpertSwiGluLaunch value) override {
    calls.push_back("swiglu"); swiglu_launch = value; return next();
  }
  Status finalize(DeepSeekExpertFinalizeLaunch value) override {
    calls.push_back("finalize"); finalize_launch = value; return next();
  }
  Status next() {
    return calls.size() == fail_at ? Status::Internal("injected") : Status::Ok();
  }
  std::size_t fail_at = 99;
  std::vector<std::string> calls;
  std::vector<DeepSeekFp8ActivationQuantLaunch> quant;
  std::vector<DeepSeekFp8GemmLaunch> gemms;
  DeepSeekSharedExpertSwiGluLaunch swiglu_launch{};
  DeepSeekExpertFinalizeLaunch finalize_launch{};
};

DeepSeekSharedExpertArena arena() {
  return {{100, 4096U * 4U}, {200, 32U * 4U}, {300, 2048U * 2U * 4U},
          {400, 2048U * 2U * 4U}, {500, 4096U * 2U * 4U}, {600, 4}};
}

TEST(DeepSeekSharedExpertDriverTest, EmitsCanonicalResidentPipeline) {
  Operations operations;
  auto driver = DeepSeekSharedExpertDriver::Create(
      4, {11, 12, 13, 14, 15, 16}, arena(), 21, 22, 23, 24,
      25, &operations.host_error, operations);
  ASSERT_TRUE(driver.ok());
  ASSERT_TRUE(driver->execute(3).ok());
  EXPECT_EQ(operations.calls,
            (std::vector<std::string>{"zero", "quant", "gemm", "gemm",
                                      "swiglu", "quant", "gemm", "finalize", "copy", "event"}));
  EXPECT_FALSE(driver->execute(3).ok());
  ASSERT_TRUE(driver->poll().ok());
  ASSERT_EQ(operations.quant.size(), 2U);
  EXPECT_EQ(operations.quant[0].logical_k, 4096U);
  EXPECT_EQ(operations.quant[1].logical_k, 2048U);
  ASSERT_EQ(operations.gemms.size(), 3U);
  EXPECT_EQ(operations.gemms[0].n, 2048U);
  EXPECT_EQ(operations.gemms[0].k, 4096U);
  EXPECT_EQ(operations.gemms[2].n, 4096U);
  EXPECT_EQ(operations.gemms[2].k, 2048U);
  EXPECT_EQ(operations.swiglu_launch.gate_bf16,
            operations.swiglu_launch.output_bf16);
  EXPECT_EQ(operations.finalize_launch.accumulator_f32, 22U);
}

TEST(DeepSeekSharedExpertDriverTest, FailsCreationForUndersizedArena) {
  Operations operations;
  auto value = arena();
  value.activation_scale_bits.bytes--;
  EXPECT_FALSE(DeepSeekSharedExpertDriver::Create(
                   4, {11, 12, 13, 14, 15, 16}, value, 21, 22, 23, 24,
                   25, &operations.host_error, operations)
                   .ok());
}

TEST(DeepSeekSharedExpertDriverTest, PoisonsAfterPartialSubmissionFailure) {
  Operations operations;
  operations.fail_at = 4;
  auto driver = DeepSeekSharedExpertDriver::Create(
      4, {11, 12, 13, 14, 15, 16}, arena(), 21, 22, 23, 24,
      25, &operations.host_error, operations);
  ASSERT_TRUE(driver.ok());
  EXPECT_FALSE(driver->execute(3).ok());
  const auto submitted = operations.calls.size();
  EXPECT_FALSE(driver->execute(3).ok());
  EXPECT_EQ(operations.calls.size(), submitted);
}

TEST(DeepSeekSharedExpertDriverTest, WaitsForCopyEventBeforeReadingDeviceError) {
  Operations operations;
  auto driver = DeepSeekSharedExpertDriver::Create(
      4, {11, 12, 13, 14, 15, 16}, arena(), 21, 22, 23, 24,
      25, &operations.host_error, operations);
  ASSERT_TRUE(driver.ok());
  EXPECT_FALSE(driver->poll().ok());
  ASSERT_TRUE(driver->execute(3).ok());
  operations.completion = DeepSeekExpertAsyncStatus::kInProgress;
  operations.host_error = 99;  // Must not be inspected while copy is pending.
  EXPECT_EQ(driver->poll().value(), DeepSeekExpertAsyncStatus::kInProgress);
  EXPECT_FALSE(driver->execute(3).ok());
  operations.completion = DeepSeekExpertAsyncStatus::kSuccess;
  operations.device_error = 2;
  EXPECT_EQ(driver->poll().value(), DeepSeekExpertAsyncStatus::kError);
  EXPECT_FALSE(driver->execute(3).ok());
  EXPECT_FALSE(driver->poll().ok());
}

TEST(DeepSeekSharedExpertDriverTest, AllowsReuseOnlyAfterSuccessfulCompletion) {
  Operations operations;
  auto driver = DeepSeekSharedExpertDriver::Create(
      4, {11, 12, 13, 14, 15, 16}, arena(), 21, 22, 23, 24,
      25, &operations.host_error, operations);
  ASSERT_TRUE(driver.ok());
  ASSERT_TRUE(driver->execute(1).ok());
  EXPECT_EQ(driver->poll().value(), DeepSeekExpertAsyncStatus::kSuccess);
  EXPECT_FALSE(driver->poll().ok());
  EXPECT_TRUE(driver->execute(4).ok());
}

TEST(DeepSeekSharedExpertDriverTest, PoisonsAfterCopyOrEventSubmissionFailure) {
  for (const auto fail_at : {9U, 10U}) {
    Operations operations;
    operations.fail_at = fail_at;
    auto driver = DeepSeekSharedExpertDriver::Create(
        4, {11, 12, 13, 14, 15, 16}, arena(), 21, 22, 23, 24,
        25, &operations.host_error, operations);
    ASSERT_TRUE(driver.ok());
    EXPECT_FALSE(driver->execute(1).ok());
    EXPECT_FALSE(driver->poll().ok());
    EXPECT_FALSE(driver->execute(1).ok());
  }
}

class RoutedStage final : public DeepSeekStageOperatorBackend {
 public:
  Status launch(const DeepSeekStageOperatorCommand&,
      const DeepSeekPipelinePlanDescriptor&) override { ++launches; return Status::Ok(); }
  Result<DeepSeekStageComputeStatus> poll() override { return completion; }
  unsigned launches = 0;
  DeepSeekStageComputeStatus completion = DeepSeekStageComputeStatus::kInProgress;
};

class SharedProvider final : public DeepSeekSharedExpertProvider {
 public:
  Status prepare(std::uint32_t, const DeepSeekPipelinePlanDescriptor&) override {
    ++prepares;
    return fail_prepare ? Status::Internal("injected preparation failure") : Status::Ok();
  }
  unsigned prepares = 0;
  bool fail_prepare = false;
  Result<DeepSeekSharedExpertDriver*> resolve(std::uint32_t,
      const DeepSeekPipelinePlanDescriptor&) override { ++resolves; return driver; }
  DeepSeekSharedExpertDriver* driver = nullptr;
  unsigned resolves = 0;
};

TEST(DeepSeekSharedExpertStageTest, CompletesOnlyAfterBothBranchesAndErrorReadback) {
  Operations operations;
  auto driver = DeepSeekSharedExpertDriver::Create(
      4, {11, 12, 13, 14, 15, 16}, arena(), 21, 22, 23, 24,
      25, &operations.host_error, operations);
  ASSERT_TRUE(driver.ok());
  SharedProvider provider;
  provider.driver = &*driver;
  RoutedStage routed;
  DeepSeekSharedExpertStageBackend backend(routed, provider);
  DeepSeekPipelinePlanDescriptor plan{};
  plan.engine_epoch = 1;
  plan.plan_sequence = 1;
  plan.token_count = 3;
  const DeepSeekStageOperatorCommand command{DeepSeekStageOperatorKind::kMoe, 2};
  ASSERT_TRUE(backend.launch(command, plan).ok());
  EXPECT_EQ(provider.prepares, 1U);
  EXPECT_EQ(backend.poll().value(), DeepSeekStageComputeStatus::kInProgress);
  EXPECT_TRUE(operations.calls.empty());
  routed.completion = DeepSeekStageComputeStatus::kSuccess;
  EXPECT_EQ(backend.poll().value(), DeepSeekStageComputeStatus::kInProgress);
  EXPECT_EQ(operations.calls.back(), "event");
  operations.completion = DeepSeekExpertAsyncStatus::kInProgress;
  EXPECT_EQ(backend.poll().value(), DeepSeekStageComputeStatus::kInProgress);
  EXPECT_FALSE(backend.launch(command, plan).ok());
  operations.completion = DeepSeekExpertAsyncStatus::kSuccess;
  EXPECT_EQ(backend.poll().value(), DeepSeekStageComputeStatus::kSuccess);
  EXPECT_FALSE(backend.poll().ok());
}

TEST(DeepSeekSharedExpertStageTest, RejectsMissingSharedBeforeLaunchingRoutedWork) {
  SharedProvider provider;
  RoutedStage routed;
  DeepSeekSharedExpertStageBackend backend(routed, provider);
  DeepSeekPipelinePlanDescriptor plan{};
  plan.engine_epoch = 1; plan.plan_sequence = 1; plan.token_count = 1;
  EXPECT_FALSE(backend.launch({DeepSeekStageOperatorKind::kMoe, 0}, plan).ok());
  EXPECT_EQ(routed.launches, 0U);
  EXPECT_FALSE(backend.poll().ok());
}

TEST(DeepSeekSharedExpertStageTest, PassesOtherOperatorsWithoutResolvingSharedWeights) {
  SharedProvider provider;
  RoutedStage routed;
  routed.completion = DeepSeekStageComputeStatus::kSuccess;
  DeepSeekSharedExpertStageBackend backend(routed, provider);
  DeepSeekPipelinePlanDescriptor plan{};
  plan.engine_epoch = 1; plan.plan_sequence = 1; plan.token_count = 1;
  ASSERT_TRUE(backend.launch({DeepSeekStageOperatorKind::kAttention, 0}, plan).ok());
  EXPECT_EQ(backend.poll().value(), DeepSeekStageComputeStatus::kSuccess);
  EXPECT_EQ(provider.resolves, 0U);
  EXPECT_EQ(provider.prepares, 0U);
}

TEST(DeepSeekSharedExpertStageTest, PreparationFailureDoesNotLaunchRoutedWork) {
  Operations operations;
  auto driver = DeepSeekSharedExpertDriver::Create(
      4, {11, 12, 13, 14, 15, 16}, arena(), 21, 22, 23, 24,
      25, &operations.host_error, operations);
  ASSERT_TRUE(driver.ok());
  SharedProvider provider;
  provider.driver = &*driver;
  provider.fail_prepare = true;
  RoutedStage routed;
  DeepSeekSharedExpertStageBackend backend(routed, provider);
  DeepSeekPipelinePlanDescriptor plan{};
  plan.engine_epoch = 1; plan.plan_sequence = 1; plan.token_count = 1;
  EXPECT_FALSE(backend.launch({DeepSeekStageOperatorKind::kMoe, 0}, plan).ok());
  EXPECT_EQ(provider.prepares, 1U);
  EXPECT_EQ(routed.launches, 0U);
  EXPECT_TRUE(operations.calls.empty());
  EXPECT_FALSE(backend.poll().ok());
  EXPECT_FALSE(backend.launch({DeepSeekStageOperatorKind::kMoe, 0}, plan).ok());
}

} }  // namespace pih
