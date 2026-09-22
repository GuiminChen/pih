#include <array>
#include <cstdint>
#include <cstring>
#include <string>

#include <gtest/gtest.h>

#include "pih/model/qwen3_bf16_dispatch_plan.h"

namespace pih {
namespace {

TensorView view(std::uintptr_t address, DType dtype,
                std::span<const std::int64_t> shape,
                std::uint64_t generation = 1,
                std::span<const std::int64_t> strides = {}) {
  return TensorView::Create(reinterpret_cast<void*>(address), dtype, shape,
                            strides,
                            Device::Create(DeviceType::kCuda, 0).value(),
                            generation)
      .value();
}

ResolvedKernelFunction function(QwenBf16Primitive primitive) {
  const std::string cubin_digest(64, 'c');
  auto manifest = qwen_bf16_kernel_manifest(primitive, cubin_digest).value();
  return {std::string(qwen_bf16_kernel_symbol(primitive).value()),
          std::string(manifest.logical_id()), cubin_digest,
          std::string(manifest.parameter_abi_sha256()), 23};
}

class RecordingLaunchDriver final : public KernelLaunchDriver {
 public:
  Status launch(DriverFunctionHandle, const KernelLaunchGeometry&,
                DriverStreamHandle, void**) override {
    ++calls;
    return result;
  }
  int calls = 0;
  Status result = Status::Ok();
};

std::uint64_t read_u64(const KernelArgumentPacket& packet,
                       std::size_t ordinal) {
  std::uint64_t value = 0;
  std::memcpy(&value, packet.argument_cell(ordinal), sizeof(value));
  return value;
}

std::uint32_t read_u32(const KernelArgumentPacket& packet,
                       std::size_t ordinal) {
  std::uint32_t value = 0;
  std::memcpy(&value, packet.argument_cell(ordinal), sizeof(value));
  return value;
}

TEST(QwenBf16DispatchPlanTest, BuildsEmbeddingPacketFromExactTensorShapes) {
  const std::array<std::int64_t, 2> table_shape{4, 3};
  const std::array<std::int64_t, 1> ids_shape{2};
  const std::array<std::int64_t, 2> output_shape{2, 3};
  auto plan = QwenBf16DispatchPlan::CreateEmbedding(
      function(QwenBf16Primitive::kEmbedding),
      view(0x1000, DType::kBFloat16, table_shape),
      view(0x2000, DType::kInt64, ids_shape),
      view(0x3000, DType::kBFloat16, output_shape), 0);
  ASSERT_TRUE(plan.ok()) << plan.status().message();
  EXPECT_TRUE(plan->arguments().ready());
  EXPECT_EQ(read_u64(plan->arguments(), 3), 2);
  EXPECT_EQ(read_u64(plan->arguments(), 4), 4);
  EXPECT_EQ(read_u64(plan->arguments(), 5), 3);
  EXPECT_EQ(plan->geometry().block_x(), 256);
  EXPECT_EQ(plan->geometry().grid_x(), 1);
}

TEST(QwenBf16DispatchPlanTest, RejectsEmbeddingShapeDtypeAndFunctionDrift) {
  const std::array<std::int64_t, 2> table_shape{4, 3};
  const std::array<std::int64_t, 1> ids_shape{2};
  const std::array<std::int64_t, 2> wrong_output_shape{2, 4};
  auto embedding = function(QwenBf16Primitive::kEmbedding);
  EXPECT_FALSE(QwenBf16DispatchPlan::CreateEmbedding(
                   embedding, view(0x1000, DType::kBFloat16, table_shape),
                   view(0x2000, DType::kInt32, ids_shape),
                   view(0x3000, DType::kBFloat16, wrong_output_shape), 0)
                   .ok());
  embedding.parameter_abi_sha256 = std::string(64, 'a');
  const std::array<std::int64_t, 2> output_shape{2, 3};
  EXPECT_FALSE(QwenBf16DispatchPlan::CreateEmbedding(
                   embedding, view(0x1000, DType::kBFloat16, table_shape),
                   view(0x2000, DType::kInt64, ids_shape),
                   view(0x3000, DType::kBFloat16, output_shape), 0)
                   .ok());
}

TEST(QwenBf16DispatchPlanTest, ElementwiseAllowsExactButNotPartialAlias) {
  const std::array<std::int64_t, 1> shape{8};
  auto exact = QwenBf16DispatchPlan::CreateElementwise(
      QwenBf16Primitive::kResidualAdd,
      function(QwenBf16Primitive::kResidualAdd),
      view(0x1000, DType::kBFloat16, shape),
      view(0x2000, DType::kBFloat16, shape),
      view(0x1000, DType::kBFloat16, shape), 0);
  ASSERT_TRUE(exact.ok()) << exact.status().message();
  EXPECT_EQ(read_u64(exact->arguments(), 3), 8);

  EXPECT_FALSE(QwenBf16DispatchPlan::CreateElementwise(
                   QwenBf16Primitive::kSiluMul,
                   function(QwenBf16Primitive::kSiluMul),
                   view(0x1000, DType::kBFloat16, shape),
                   view(0x2000, DType::kBFloat16, shape),
                   view(0x1002, DType::kBFloat16, shape), 0)
                   .ok());
}

TEST(QwenBf16DispatchPlanTest, SubmissionIsOneShotEvenAfterDriverFailure) {
  const std::array<std::int64_t, 1> shape{8};
  auto plan = QwenBf16DispatchPlan::CreateElementwise(
      QwenBf16Primitive::kResidualAdd,
      function(QwenBf16Primitive::kResidualAdd),
      view(0x1000, DType::kBFloat16, shape),
      view(0x2000, DType::kBFloat16, shape),
      view(0x3000, DType::kBFloat16, shape), 0);
  ASSERT_TRUE(plan.ok());
  RecordingLaunchDriver driver;
  driver.result = Status::Internal("injected launch failure");
  EXPECT_FALSE(plan->submit(driver, 41).ok());
  EXPECT_TRUE(plan->submitted());
  EXPECT_EQ(driver.calls, 1);
  EXPECT_FALSE(plan->submit(driver, 41).ok());
  EXPECT_EQ(driver.calls, 1);
}

TEST(QwenBf16DispatchPlanTest, RejectsNoncontiguousAndWrongPrimitive) {
  const std::array<std::int64_t, 2> shape{2, 4};
  const std::array<std::int64_t, 2> strides{5, 1};
  EXPECT_FALSE(QwenBf16DispatchPlan::CreateElementwise(
                   QwenBf16Primitive::kResidualAdd,
                   function(QwenBf16Primitive::kResidualAdd),
                   view(0x1000, DType::kBFloat16, shape, 1, strides),
                   view(0x2000, DType::kBFloat16, shape),
                   view(0x3000, DType::kBFloat16, shape), 0)
                   .ok());
  EXPECT_FALSE(QwenBf16DispatchPlan::CreateElementwise(
                   QwenBf16Primitive::kEmbedding,
                   function(QwenBf16Primitive::kEmbedding),
                   view(0x1000, DType::kBFloat16, shape),
                   view(0x2000, DType::kBFloat16, shape),
                   view(0x3000, DType::kBFloat16, shape), 0)
                   .ok());
}

TEST(QwenBf16DispatchPlanTest, BuildsFixedTreeRmsNormAndAllowsInPlaceOutput) {
  const std::array<std::int64_t, 2> activation_shape{3, 1024};
  const std::array<std::int64_t, 1> weight_shape{1024};
  auto plan = QwenBf16DispatchPlan::CreateRmsNorm(
      function(QwenBf16Primitive::kRmsNorm),
      view(0x10000, DType::kBFloat16, activation_shape),
      view(0x20000, DType::kBFloat16, weight_shape),
      view(0x10000, DType::kBFloat16, activation_shape), 1.0e-6F, 0);
  ASSERT_TRUE(plan.ok()) << plan.status().message();
  EXPECT_EQ(plan->geometry().grid_x(), 3);
  EXPECT_EQ(plan->geometry().block_x(), 256);
  EXPECT_EQ(read_u64(plan->arguments(), 3), 3);
  EXPECT_EQ(read_u64(plan->arguments(), 4), 1024);
}

TEST(QwenBf16DispatchPlanTest, BuildsHeadwiseRmsNormForQueryAndKey) {
  const std::array<std::int64_t, 3> query_shape{2, 16, 128};
  const std::array<std::int64_t, 1> weight_shape{128};
  auto plan = QwenBf16DispatchPlan::CreateRmsNorm(
      function(QwenBf16Primitive::kRmsNorm),
      view(0x10000, DType::kBFloat16, query_shape),
      view(0x20000, DType::kBFloat16, weight_shape),
      view(0x10000, DType::kBFloat16, query_shape), 1.0e-6F, 0);
  ASSERT_TRUE(plan.ok()) << plan.status().message();
  EXPECT_EQ(plan->geometry().grid_x(), 32);
  EXPECT_EQ(read_u64(plan->arguments(), 3), 32);
  EXPECT_EQ(read_u64(plan->arguments(), 4), 128);
}

TEST(QwenBf16DispatchPlanTest, RejectsRmsNormShapeEpsilonAndWeightAlias) {
  const std::array<std::int64_t, 2> wrong_shape{2, 768};
  const std::array<std::int64_t, 2> activation_shape{2, 1024};
  const std::array<std::int64_t, 1> weight_shape{1024};
  const auto rms = function(QwenBf16Primitive::kRmsNorm);
  EXPECT_FALSE(QwenBf16DispatchPlan::CreateRmsNorm(
                   rms, view(0x10000, DType::kBFloat16, wrong_shape),
                   view(0x20000, DType::kBFloat16, weight_shape),
                   view(0x30000, DType::kBFloat16, wrong_shape), 1.0e-6F, 0)
                   .ok());
  EXPECT_FALSE(QwenBf16DispatchPlan::CreateRmsNorm(
                   rms, view(0x10000, DType::kBFloat16, activation_shape),
                   view(0x20000, DType::kBFloat16, weight_shape),
                   view(0x30000, DType::kBFloat16, activation_shape), 0.0F, 0)
                   .ok());
  EXPECT_FALSE(QwenBf16DispatchPlan::CreateRmsNorm(
                   rms, view(0x10000, DType::kBFloat16, activation_shape),
                   view(0x10000, DType::kBFloat16, weight_shape),
                   view(0x30000, DType::kBFloat16, activation_shape), 1.0e-6F,
                   0)
                   .ok());
}

TEST(QwenBf16DispatchPlanTest, BuildsRotateHalfRopeWithExactInputAlias) {
  const std::array<std::int64_t, 3> input_shape{2, 16, 128};
  const std::array<std::int64_t, 2> angle_shape{2, 64};
  auto plan = QwenBf16DispatchPlan::CreateRope(
      function(QwenBf16Primitive::kRope),
      view(0x10000, DType::kBFloat16, input_shape),
      view(0x20000, DType::kFloat32, angle_shape),
      view(0x30000, DType::kFloat32, angle_shape),
      view(0x10000, DType::kBFloat16, input_shape), 0);
  ASSERT_TRUE(plan.ok()) << plan.status().message();
  EXPECT_EQ(read_u64(plan->arguments(), 4), 32);
  EXPECT_EQ(read_u64(plan->arguments(), 5), 16);
  EXPECT_EQ(read_u64(plan->arguments(), 6), 128);
  EXPECT_EQ(plan->geometry().block_x(), 256);
  EXPECT_EQ(plan->geometry().grid_x(), 8);
}

TEST(QwenBf16DispatchPlanTest, RejectsRopeAngleAndHeadDimensionDrift) {
  const std::array<std::int64_t, 3> input_shape{2, 16, 128};
  const std::array<std::int64_t, 3> wrong_head{2, 16, 64};
  const std::array<std::int64_t, 3> wrong_heads{2, 4, 128};
  const std::array<std::int64_t, 1> short_angles{127};
  const std::array<std::int64_t, 2> angle_shape{2, 64};
  const auto rope = function(QwenBf16Primitive::kRope);
  EXPECT_FALSE(QwenBf16DispatchPlan::CreateRope(
                   rope, view(0x10000, DType::kBFloat16, wrong_head),
                   view(0x20000, DType::kFloat32, angle_shape),
                   view(0x30000, DType::kFloat32, angle_shape),
                   view(0x40000, DType::kBFloat16, wrong_head), 0)
                   .ok());
  EXPECT_FALSE(QwenBf16DispatchPlan::CreateRope(
                   rope, view(0x10000, DType::kBFloat16, input_shape),
                   view(0x20000, DType::kFloat32, short_angles),
                   view(0x30000, DType::kFloat32, short_angles),
                   view(0x40000, DType::kBFloat16, input_shape), 0)
                   .ok());
  EXPECT_FALSE(QwenBf16DispatchPlan::CreateRope(
                   rope, view(0x10000, DType::kBFloat16, wrong_heads),
                   view(0x20000, DType::kFloat32, angle_shape),
                   view(0x30000, DType::kFloat32, angle_shape),
                   view(0x40000, DType::kBFloat16, wrong_heads), 0)
                   .ok());
}

TEST(QwenBf16DispatchPlanTest, BuildsSharedRopeAngleGenerationPacket) {
  const std::array<std::int64_t, 1> positions_shape{17};
  const std::array<std::int64_t, 2> angles_shape{17, 64};
  auto plan = QwenBf16DispatchPlan::CreateRopeAngles(
      function(QwenBf16Primitive::kRopeAngles),
      view(0x10000, DType::kInt64, positions_shape),
      view(0x20000, DType::kFloat32, angles_shape),
      view(0x30000, DType::kFloat32, angles_shape), 0);
  ASSERT_TRUE(plan.ok()) << plan.status().message();
  EXPECT_EQ(read_u64(plan->arguments(), 3), 17);
  EXPECT_EQ(read_u64(plan->arguments(), 4), 128);
  EXPECT_EQ(plan->geometry().grid_x(), 5);
}

TEST(QwenBf16DispatchPlanTest, BuildsFp32LowestTieArgmaxPacket) {
  const std::array<std::int64_t, 2> logits_shape{1, 151936};
  const std::array<std::int64_t, 1> sampled_shape{1};
  const std::array<std::int64_t, 1> error_shape{4};
  auto plan = QwenBf16DispatchPlan::CreateGreedyArgmax(
      function(QwenBf16Primitive::kGreedyArgmax),
      view(0x10000, DType::kFloat32, logits_shape),
      view(0x100000, DType::kInt64, sampled_shape),
      view(0x110000, DType::kUInt8, error_shape), 0);
  ASSERT_TRUE(plan.ok()) << plan.status().message();
  EXPECT_EQ(read_u64(plan->arguments(), 3), 151936);
  EXPECT_EQ(plan->geometry().grid_x(), 1);
  EXPECT_EQ(plan->geometry().block_x(), 256);
}

TEST(QwenBf16DispatchPlanTest, RejectsAngleAndArgmaxDtypeDrift) {
  const std::array<std::int64_t, 1> positions_shape{2};
  const std::array<std::int64_t, 2> angles_shape{2, 64};
  EXPECT_FALSE(QwenBf16DispatchPlan::CreateRopeAngles(
                   function(QwenBf16Primitive::kRopeAngles),
                   view(0x10000, DType::kInt32, positions_shape),
                   view(0x20000, DType::kFloat32, angles_shape),
                   view(0x30000, DType::kFloat32, angles_shape), 0)
                   .ok());
  const std::array<std::int64_t, 2> logits_shape{1, 151936};
  const std::array<std::int64_t, 1> sampled_shape{1};
  const std::array<std::int64_t, 1> error_shape{4};
  EXPECT_FALSE(QwenBf16DispatchPlan::CreateGreedyArgmax(
                   function(QwenBf16Primitive::kGreedyArgmax),
                   view(0x10000, DType::kBFloat16, logits_shape),
                   view(0x100000, DType::kInt64, sampled_shape),
                   view(0x110000, DType::kUInt8, error_shape), 0)
                   .ok());
}

TEST(QwenBf16DispatchPlanTest, BuildsBoundedKvAppendPacket) {
  const std::array<std::int64_t, 3> activation_shape{2, 8, 128};
  const std::array<std::int64_t, 1> kv_shape{1'835'008};
  const std::array<std::int64_t, 1> states_shape{16};
  const std::array<std::int64_t, 1> handles_shape{16};
  const std::array<std::int64_t, 1> offsets_shape{4};
  const std::array<std::int64_t, 1> error_shape{4};
  auto plan = QwenBf16DispatchPlan::CreateKvAppend(
      function(QwenBf16Primitive::kKvAppend),
      view(0x10000, DType::kBFloat16, activation_shape),
      view(0x20000, DType::kBFloat16, activation_shape),
      view(0x100000, DType::kUInt8, kv_shape),
      view(0x400000, DType::kUInt8, states_shape),
      view(0x410000, DType::kUInt8, handles_shape),
      view(0x420000, DType::kUInt8, offsets_shape),
      view(0x430000, DType::kUInt8, error_shape), 17, 27, 1, 0);
  ASSERT_TRUE(plan.ok()) << plan.status().message();
  EXPECT_TRUE(plan->arguments().ready());
  EXPECT_EQ(read_u32(plan->arguments(), 7), 17);
  EXPECT_EQ(read_u32(plan->arguments(), 8), 27);
  EXPECT_EQ(read_u64(plan->arguments(), 9), 2);
  EXPECT_EQ(read_u32(plan->arguments(), 10), 1);
  EXPECT_EQ(plan->geometry().block_x(), 256);
  EXPECT_EQ(plan->geometry().grid_x(), 16);
}

TEST(QwenBf16DispatchPlanTest, RejectsKvAppendExtentAndAliasDrift) {
  const std::array<std::int64_t, 3> activation_shape{1, 8, 128};
  const std::array<std::int64_t, 1> short_kv_shape{1'835'007};
  const std::array<std::int64_t, 1> kv_shape{1'835'008};
  const std::array<std::int64_t, 1> states_shape{16};
  const std::array<std::int64_t, 1> handles_shape{8};
  const std::array<std::int64_t, 1> offsets_shape{2};
  const std::array<std::int64_t, 1> error_shape{4};
  const auto append = function(QwenBf16Primitive::kKvAppend);
  EXPECT_FALSE(QwenBf16DispatchPlan::CreateKvAppend(
                   append, view(0x10000, DType::kBFloat16, activation_shape),
                   view(0x20000, DType::kBFloat16, activation_shape),
                   view(0x100000, DType::kUInt8, short_kv_shape),
                   view(0x400000, DType::kUInt8, states_shape),
                   view(0x410000, DType::kUInt8, handles_shape),
                   view(0x420000, DType::kUInt8, offsets_shape),
                   view(0x430000, DType::kUInt8, error_shape), 1, 0, 1, 0)
                   .ok());
  EXPECT_FALSE(QwenBf16DispatchPlan::CreateKvAppend(
                   append, view(0x10000, DType::kBFloat16, activation_shape),
                   view(0x10000, DType::kBFloat16, activation_shape),
                   view(0x100000, DType::kUInt8, kv_shape),
                   view(0x400000, DType::kUInt8, states_shape),
                   view(0x410000, DType::kUInt8, handles_shape),
                   view(0x420000, DType::kUInt8, offsets_shape),
                   view(0x430000, DType::kUInt8, error_shape), 1, 0, 1, 0)
                   .ok());
}

TEST(QwenBf16DispatchPlanTest, BuildsMultiRowCausalPagedGqaPacket) {
  const std::array<std::int64_t, 3> activation_shape{2, 16, 128};
  const std::array<std::int64_t, 1> kv_shape{3'670'016};
  const std::array<std::int64_t, 1> states_shape{32};
  const std::array<std::int64_t, 1> handles_shape{16};
  const std::array<std::int64_t, 1> error_shape{4};
  auto plan = QwenBf16DispatchPlan::CreatePagedGqa(
      function(QwenBf16Primitive::kPagedGqa),
      view(0x10000, DType::kBFloat16, activation_shape),
      view(0x20000, DType::kBFloat16, activation_shape),
      view(0x100000, DType::kUInt8, kv_shape),
      view(0x600000, DType::kUInt8, states_shape),
      view(0x610000, DType::kUInt8, handles_shape),
      view(0x620000, DType::kUInt8, error_shape), 23, 27, 15, 17,
      0.0883883476F, 2, 0);
  ASSERT_TRUE(plan.ok()) << plan.status().message();
  EXPECT_TRUE(plan->arguments().ready());
  EXPECT_EQ(read_u32(plan->arguments(), 6), 23);
  EXPECT_EQ(read_u32(plan->arguments(), 7), 27);
  EXPECT_EQ(read_u64(plan->arguments(), 8), 15);
  EXPECT_EQ(read_u32(plan->arguments(), 9), 2);
  EXPECT_EQ(read_u32(plan->arguments(), 10), 2);
  EXPECT_EQ(read_u32(plan->arguments(), 11), 17);
  EXPECT_EQ(read_u32(plan->arguments(), 13), 2);
  EXPECT_EQ(plan->geometry().grid_x(), 32);
  EXPECT_EQ(plan->geometry().block_x(), 1);
}

TEST(QwenBf16DispatchPlanTest, RejectsPagedGqaCausalAndHandleDrift) {
  const std::array<std::int64_t, 3> activation_shape{1, 16, 128};
  const std::array<std::int64_t, 1> kv_shape{1'835'008};
  const std::array<std::int64_t, 1> states_shape{16};
  const std::array<std::int64_t, 1> short_handles_shape{8};
  const std::array<std::int64_t, 1> exact_handles_shape{16};
  const std::array<std::int64_t, 1> error_shape{4};
  const auto gqa = function(QwenBf16Primitive::kPagedGqa);
  EXPECT_FALSE(QwenBf16DispatchPlan::CreatePagedGqa(
                   gqa, view(0x10000, DType::kBFloat16, activation_shape),
                   view(0x20000, DType::kBFloat16, activation_shape),
                   view(0x100000, DType::kUInt8, kv_shape),
                   view(0x400000, DType::kUInt8, states_shape),
                   view(0x410000, DType::kUInt8, short_handles_shape),
                   view(0x420000, DType::kUInt8, error_shape), 1, 0, 16, 17,
                   0.1F, 1, 0)
                   .ok());
  EXPECT_FALSE(QwenBf16DispatchPlan::CreatePagedGqa(
                   gqa, view(0x10000, DType::kBFloat16, activation_shape),
                   view(0x20000, DType::kBFloat16, activation_shape),
                   view(0x100000, DType::kUInt8, kv_shape),
                   view(0x400000, DType::kUInt8, states_shape),
                   view(0x410000, DType::kUInt8, exact_handles_shape),
                   view(0x420000, DType::kUInt8, error_shape), 1, 0, 17, 17,
                   0.1F, 1, 0)
                   .ok());
}

}  // namespace
}  // namespace pih
