#include <array>
#include <cstdint>
#include <cstring>
#include <string>

#include <gtest/gtest.h>

#include "pih/model/qwen3_teacher_forced_metric_plan.h"

namespace pih {
namespace {

TensorView view(std::uintptr_t address, DType dtype,
                std::span<const std::int64_t> shape,
                std::uint64_t generation = 1) {
  return TensorView::Create(reinterpret_cast<void*>(address), dtype, shape, {},
                            Device::Create(DeviceType::kCuda, 0).value(),
                            generation).value();
}

ResolvedKernelFunction function() {
  const std::string digest(64, 'c');
  auto manifest = qwen_bf16_kernel_manifest(
      QwenBf16Primitive::kTeacherForcedMetric, digest).value();
  return {std::string(qwen_bf16_kernel_symbol(
              QwenBf16Primitive::kTeacherForcedMetric).value()),
          std::string(manifest.logical_id()), digest,
          std::string(manifest.parameter_abi_sha256()), 73};
}

class Driver final : public KernelLaunchDriver {
 public:
  Status launch(DriverFunctionHandle function,
                const KernelLaunchGeometry& geometry,
                DriverStreamHandle stream, void**) override {
    ++calls; observed_function = function; observed_grid = geometry.grid_x();
    observed_stream = stream; return result;
  }
  int calls = 0;
  DriverFunctionHandle observed_function = 0;
  std::uint32_t observed_grid = 0;
  DriverStreamHandle observed_stream = 0;
  Status result = Status::Ok();
};

std::uint32_t u32(const KernelArgumentPacket& packet, std::size_t ordinal) {
  std::uint32_t value = 0;
  std::memcpy(&value, packet.argument_cell(ordinal), sizeof(value));
  return value;
}

TEST(QwenTeacherForcedMetricPlanTest, FreezesExactShapesPacketAndGeometry) {
  constexpr std::int64_t rows = 17;
  const std::array<std::int64_t, 2> logits_shape{rows, 151936};
  const std::array<std::int64_t, 1> vector_shape{rows};
  const std::array<std::int64_t, 1> error_shape{1};
  auto plan = QwenTeacherForcedMetricPlan::Create(
      function(), view(0x100000, DType::kFloat32, logits_shape),
      view(0xb00000, DType::kUInt32, vector_shape),
      view(0xb10000, DType::kUInt32, vector_shape),
      view(0xb20000, DType::kFloat64, vector_shape),
      view(0xb30000, DType::kUInt32, vector_shape),
      view(0xb40000, DType::kUInt32, error_shape), 0);
  ASSERT_TRUE(plan.ok()) << plan.status().message();
  EXPECT_EQ(plan->rows(), 17U);
  EXPECT_EQ(plan->geometry().grid_x(), 17U);
  EXPECT_EQ(plan->geometry().block_x(), 256U);
  EXPECT_TRUE(plan->arguments().ready());
  EXPECT_EQ(u32(plan->arguments(), 6), 17U);
  EXPECT_EQ(u32(plan->arguments(), 7), 151936U);
}

TEST(QwenTeacherForcedMetricPlanTest, RejectsShapeDtypeGenerationAndAbiDrift) {
  const std::array<std::int64_t, 2> logits_shape{2, 151936};
  const std::array<std::int64_t, 1> vector_shape{2};
  const std::array<std::int64_t, 1> error_shape{1};
  auto build = [&](ResolvedKernelFunction resolved, DType nll_dtype,
                   std::uint64_t generation) {
    return QwenTeacherForcedMetricPlan::Create(
        resolved, view(0x100000, DType::kFloat32, logits_shape),
        view(0x300000, DType::kUInt32, vector_shape),
        view(0x310000, DType::kUInt32, vector_shape),
        view(0x320000, nll_dtype, vector_shape),
        view(0x330000, DType::kUInt32, vector_shape, generation),
        view(0x340000, DType::kUInt32, error_shape), 0);
  };
  EXPECT_FALSE(build(function(), DType::kFloat32, 1).ok());
  EXPECT_FALSE(build(function(), DType::kFloat64, 0).ok());
  auto drifted = function(); drifted.parameter_abi_sha256.assign(64, 'a');
  EXPECT_FALSE(build(drifted, DType::kFloat64, 1).ok());
}

TEST(QwenTeacherForcedMetricPlanTest, SubmissionIsOneShotAfterFailure) {
  const std::array<std::int64_t, 2> logits_shape{1, 151936};
  const std::array<std::int64_t, 1> shape{1};
  auto plan = QwenTeacherForcedMetricPlan::Create(
      function(), view(0x100000, DType::kFloat32, logits_shape),
      view(0x200000, DType::kUInt32, shape),
      view(0x210000, DType::kUInt32, shape),
      view(0x220000, DType::kFloat64, shape),
      view(0x230000, DType::kUInt32, shape),
      view(0x240000, DType::kUInt32, shape), 0);
  ASSERT_TRUE(plan.ok());
  Driver driver; driver.result = Status::Internal("injected");
  EXPECT_FALSE(plan->submit(driver, 91).ok());
  EXPECT_TRUE(plan->submitted());
  EXPECT_EQ(driver.calls, 1);
  EXPECT_EQ(driver.observed_function, 73U);
  EXPECT_EQ(driver.observed_grid, 1U);
  EXPECT_EQ(driver.observed_stream, 91U);
  EXPECT_FALSE(plan->submit(driver, 91).ok());
  EXPECT_EQ(driver.calls, 1);
}

}  // namespace
}  // namespace pih
