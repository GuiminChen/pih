#include <array>
#include <cstdint>
#include <cstring>

#include <gtest/gtest.h>

#include "pih/model/qwen3_int4_dispatch_plan.h"

namespace pih {
namespace {

TensorView cuda_view(std::uintptr_t address, DType dtype,
                     std::span<const std::int64_t> shape,
                     std::uint64_t generation) {
  return TensorView::Create(reinterpret_cast<void*>(address), dtype, shape, {},
                            Device::Create(DeviceType::kCuda, 0).value(),
                            generation).value();
}

ResolvedKernelFunction resolved(const QwenInt4GemmPlan& plan) {
  const std::string cubin(64, 'c');
  const auto signature = plan.signature(cubin).value();
  return {std::string(plan.kernel_symbol()), std::string(plan.logical_id()),
          cubin, std::string(signature.parameter_abi_sha256()), 71};
}

std::uint64_t read_u64(const KernelArgumentPacket& packet,
                       std::size_t ordinal) {
  std::uint64_t value = 0;
  std::memcpy(&value, packet.argument_cell(ordinal), sizeof(value));
  return value;
}

class LaunchDriver final : public KernelLaunchDriver {
 public:
  Status launch(DriverFunctionHandle function,
                const KernelLaunchGeometry& geometry,
                DriverStreamHandle stream, void**) override {
    ++calls; last_function = function; last_grid = geometry.grid_x();
    last_stream = stream; return result;
  }
  int calls = 0;
  DriverFunctionHandle last_function = 0;
  std::uint32_t last_grid = 0;
  DriverStreamHandle last_stream = 0;
  Status result = Status::Ok();
};

TEST(QwenInt4DispatchPlanTest, MaterializesExactTypedPacketAndGeometry) {
  auto gemm = QwenInt4GemmPlan::Create(
      QwenInt4LinearShapeFamily::kQProj, 3).value();
  const std::array<std::int64_t,2> input_shape{3,1024};
  const std::array<std::int64_t,2> packed_shape{2048,512};
  const std::array<std::int64_t,2> scale_shape{2048,8};
  const std::array<std::int64_t,2> output_shape{3,2048};
  const std::array<std::int64_t,1> error_shape{4};
  auto dispatch = QwenInt4DispatchPlan::Create(
      gemm, resolved(gemm),
      cuda_view(0x10000,DType::kBFloat16,input_shape,7),
      cuda_view(0x20000,DType::kUInt8,packed_shape,11),
      cuda_view(0x120000,DType::kFloat16,scale_shape,11),
      cuda_view(0x130000,DType::kBFloat16,output_shape,7),
      cuda_view(0x140000,DType::kUInt8,error_shape,7), 0, 7, 11);
  ASSERT_TRUE(dispatch.ok()) << dispatch.status().message();
  EXPECT_TRUE(dispatch->arguments().ready());
  EXPECT_EQ(read_u64(dispatch->arguments(),0),0x10000U);
  EXPECT_EQ(read_u64(dispatch->arguments(),1),0x20000U);
  EXPECT_EQ(read_u64(dispatch->arguments(),5),3U);
  EXPECT_EQ(read_u64(dispatch->arguments(),6),2048U);
  EXPECT_EQ(read_u64(dispatch->arguments(),7),1024U);
  EXPECT_EQ(dispatch->geometry().grid_x(),24U);
  EXPECT_EQ(dispatch->geometry().block_x(),256U);
}

TEST(QwenInt4DispatchPlanTest, RejectsShapeGenerationAndFunctionDrift) {
  auto gemm = QwenInt4GemmPlan::Create(
      QwenInt4LinearShapeFamily::kQProj, 1).value();
  const std::array<std::int64_t,2> input_shape{1,1024};
  const std::array<std::int64_t,2> wrong_packed{2048,511};
  const std::array<std::int64_t,2> packed_shape{2048,512};
  const std::array<std::int64_t,2> scale_shape{2048,8};
  const std::array<std::int64_t,2> output_shape{1,2048};
  const std::array<std::int64_t,1> error_shape{4};
  auto function = resolved(gemm);
  auto make = [&](const auto& packed, std::uint64_t activation_generation,
                  const ResolvedKernelFunction& candidate) {
    return QwenInt4DispatchPlan::Create(
        gemm,candidate,cuda_view(0x10000,DType::kBFloat16,input_shape,7),
        cuda_view(0x20000,DType::kUInt8,packed,11),
        cuda_view(0x120000,DType::kFloat16,scale_shape,11),
        cuda_view(0x130000,DType::kBFloat16,output_shape,7),
        cuda_view(0x140000,DType::kUInt8,error_shape,7),0,
        activation_generation,11);
  };
  EXPECT_FALSE(make(wrong_packed,7,function).ok());
  EXPECT_FALSE(make(packed_shape,8,function).ok());
  function.parameter_abi_sha256 = std::string(64,'a');
  EXPECT_FALSE(make(packed_shape,7,function).ok());
}

TEST(QwenInt4DispatchPlanTest, SubmissionIsOneShotAfterDriverFailure) {
  auto gemm = QwenInt4GemmPlan::Create(
      QwenInt4LinearShapeFamily::kKvProj, 1).value();
  const std::array<std::int64_t,2> input_shape{1,1024};
  const std::array<std::int64_t,2> packed_shape{1024,512};
  const std::array<std::int64_t,2> scale_shape{1024,8};
  const std::array<std::int64_t,2> output_shape{1,1024};
  const std::array<std::int64_t,1> error_shape{4};
  auto dispatch = QwenInt4DispatchPlan::Create(
      gemm,resolved(gemm),cuda_view(0x10000,DType::kBFloat16,input_shape,7),
      cuda_view(0x20000,DType::kUInt8,packed_shape,11),
      cuda_view(0xa0000,DType::kFloat16,scale_shape,11),
      cuda_view(0xb0000,DType::kBFloat16,output_shape,7),
      cuda_view(0xc0000,DType::kUInt8,error_shape,7),0,7,11);
  ASSERT_TRUE(dispatch.ok());
  LaunchDriver driver; driver.result=Status::Internal("injected");
  EXPECT_FALSE(dispatch->submit(driver,19).ok());
  EXPECT_TRUE(dispatch->submitted()); EXPECT_EQ(driver.calls,1);
  EXPECT_FALSE(dispatch->submit(driver,19).ok()); EXPECT_EQ(driver.calls,1);
}

TEST(QwenInt4DispatchPlanTest, KeepsLogicalRankSeparateFromLocalDeviceIndex) {
  auto gemm = QwenInt4GemmPlan::Create(
      QwenInt4LinearShapeFamily::kKvProj, 1).value();
  const std::array<std::int64_t,2> input_shape{1,1024};
  const std::array<std::int64_t,2> packed_shape{1024,512};
  const std::array<std::int64_t,2> scale_shape{1024,8};
  const std::array<std::int64_t,2> output_shape{1,1024};
  const std::array<std::int64_t,1> error_shape{4};
  auto dispatch = QwenInt4DispatchPlan::Create(
      gemm,resolved(gemm),cuda_view(0x10000,DType::kBFloat16,input_shape,7),
      cuda_view(0x20000,DType::kUInt8,packed_shape,11),
      cuda_view(0xa0000,DType::kFloat16,scale_shape,11),
      cuda_view(0xb0000,DType::kBFloat16,output_shape,7),
      cuda_view(0xc0000,DType::kUInt8,error_shape,7),3,7,11);
  ASSERT_TRUE(dispatch.ok()) << dispatch.status().message();
}

}  // namespace
}  // namespace pih
