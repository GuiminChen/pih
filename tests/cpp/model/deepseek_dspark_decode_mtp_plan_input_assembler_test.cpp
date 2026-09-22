#include "pih/model/deepseek_dspark_decode_mtp_plan_input_assembler.h"

#include <gtest/gtest.h>

#include <array>
#include <string>

#include "deepseek_dspark_weight_test_fixture.h"
#include "pih/model/deepseek_dspark_resident_subwave_executor.h"

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

class Operation final : public DeepSeekDsparkMtpStageOperation {
 public:
  Status launch(const DeepSeekPipelinePlanDescriptor&,
                const DeepSeekDsparkMtpStageResources&) override {
    return Status::Ok();
  }
  Status launch_prefill(
      const DeepSeekPipelinePlanDescriptor&,
      const DeepSeekDsparkMtpPrefillResources&) override {
    return Status::Ok();
  }
  Result<DeepSeekStageComputeStatus> poll() override {
    return DeepSeekStageComputeStatus::kSuccess;
  }
  Status cancel() override { return Status::Ok(); }
};

class Kernel final : public DeepSeekDsparkExpertKernelDriver {
 public:
  Status launch_resident(DeepSeekDsparkStageId, std::uint16_t,
                         std::uint64_t,
                         const DeepSeekExpertBundleDeviceView&,
                         const DeepSeekExpertRoute*,
                         std::uint32_t) override {
    return Status::Ok();
  }
  Result<DeepSeekExpertAsyncStatus> poll() override {
    return DeepSeekExpertAsyncStatus::kSuccess;
  }
};

struct Fixture final {
  Fixture()
      : weights(test::deepseek_dspark_test_weights()),
        experts(DeepSeekDsparkResidentExpertBindings::Resolve(
                    test::deepseek_dspark_test_stage(), resident_tensor)
                    .value()) {
    input.weights = &weights;
    input.resident_experts = &experts;
    input.state_layout = &state_layout;
    input.device = {
        0x1001, 0x1002, 0x1003, 0x1004, 0x1005, 0x1006,
        0x1007, 0x1008, 0x1009, 0x100a, 0x100b, 0x100c,
        0x100d, 0x100e, 0x100f, 0x1010, 0x1011, 0x1012};
    input.attention_workspace = {
        0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006,
        0x2007, 0x2008, 0x2009, 0x200a, 0x200b, 0x200c,
        0x200d, 0x200e, 0x200f, 0x2010, 0x2011, 0x2012};
    input.mhc_workspace = {
        0x3001, 0x3002, 0x3003, 0x3004, 0x3005, 0x3006};
    input.rope_frequencies_f32 = 0x4001;
    input.router_scores_f32 = 0x4002;
    input.router_host_scores = scores;
    input.router_host_bias = bias;
    auto layout = DeepSeekExpertComputeArenaLayout::Create(5).value();
    input.expert_arena =
        layout.bind(0x10000000U, layout.required_bytes()).value();
    input.expert_accumulator_f32 = 0x4003;
    input.expert_kernel = &kernel;
    for (std::size_t index = 0; index < kDeepSeekDsparkStageCount; ++index) {
      input.attention_operations[index] = &attention[index];
      input.moe_operations[index] = &moe[index];
    }
    input.stream = 0x4004;
    input.completion_event = 0x4005;
    input.current_position = 17;
    input.position_table_count = 4096;
  }

  DeepSeekDsparkWeightBindings weights;
  DeepSeekDsparkResidentExpertBindings experts;
  DeepSeekFixedStateLayout state_layout =
      DeepSeekFixedStateLayout::Build({}, true).value();
  std::array<float, 5 * 256> scores{};
  std::array<float, 256> bias{};
  std::array<Operation, kDeepSeekDsparkStageCount> attention;
  std::array<Operation, kDeepSeekDsparkStageCount> moe;
  Kernel kernel;
  DeepSeekDsparkDecodeMtpPlanInput input;
};

TEST(DeepSeekDsparkDecodeMtpPlanInputAssemblerTest,
     PublishesThreeStageResidualRingAndCompleteExecutionIdentity) {
  Fixture fixture;
  auto work = DeepSeekDsparkDecodeMtpPlanInputAssembler::Assemble(
      fixture.input);
  ASSERT_TRUE(work.ok()) << work.status().message();
  ASSERT_EQ(work->size(), kDeepSeekDsparkStageCount);
  const std::array<std::uintptr_t, 3> expected_inputs{
      fixture.input.device.draft_input_hc_bf16,
      fixture.input.device.stage_residual_a_bf16,
      fixture.input.device.stage_residual_b_bf16};
  const std::array<std::uintptr_t, 3> expected_outputs{
      fixture.input.device.stage_residual_a_bf16,
      fixture.input.device.stage_residual_b_bf16,
      fixture.input.device.draft_input_hc_bf16};
  for (std::size_t index = 0; index < work->size(); ++index) {
    const auto stage = static_cast<DeepSeekDsparkStageId>(index);
    const auto& item = work->at(index);
    EXPECT_EQ(item.attention, &fixture.attention[index]);
    EXPECT_EQ(item.moe, &fixture.moe[index]);
    EXPECT_EQ(item.resources.stage, stage);
    EXPECT_EQ(item.resources.weights, &fixture.weights.common(stage));
    EXPECT_EQ(item.resources.stage_input_hc_bf16,
              expected_inputs[index]);
    EXPECT_EQ(item.resources.attention_output_hc_bf16,
              fixture.input.mhc_workspace.residual_b_bf16);
    EXPECT_EQ(item.resources.stage_output_hc_bf16,
              expected_outputs[index]);
    EXPECT_EQ(item.resources.main_positions_u32,
              fixture.input.attention_workspace.positions_u32);
    EXPECT_EQ(item.resources.router_host_scores.data(),
              fixture.scores.data());
    EXPECT_EQ(item.resources.router_host_bias.data(), fixture.bias.data());
    EXPECT_EQ(item.resources.expert_kernel, &fixture.kernel);
    EXPECT_EQ(item.resources.transaction, nullptr);
    EXPECT_EQ(item.resources.prepare_epoch, 0U);
    EXPECT_EQ(item.resources.recent_state.recent_bf16.address, 0U);
  }
}

TEST(DeepSeekDsparkDecodeMtpPlanInputAssemblerTest,
     RejectsFuturePositionOverflowAndAliasedStageOperations) {
  Fixture overflow;
  overflow.input.current_position = 4091;
  EXPECT_EQ(DeepSeekDsparkDecodeMtpPlanInputAssembler::Assemble(
                overflow.input).status().code(),
            StatusCode::kInvalidArgument);

  Fixture aliased;
  aliased.input.moe_operations[1] = aliased.input.attention_operations[1];
  EXPECT_EQ(DeepSeekDsparkDecodeMtpPlanInputAssembler::Assemble(
                aliased.input).status().code(),
            StatusCode::kFailedPrecondition);

  Fixture buffers;
  buffers.input.mhc_workspace.residual_b_bf16 =
      buffers.input.device.stage_residual_a_bf16;
  EXPECT_EQ(DeepSeekDsparkDecodeMtpPlanInputAssembler::Assemble(
                buffers.input).status().code(),
            StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace pih
