#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "pih/backend/cuda/verified_kernel_launcher.h"

namespace pih {
namespace {

class FakeLaunchDriver final : public KernelLaunchDriver {
 public:
  Status launch(DriverFunctionHandle function,
                const KernelLaunchGeometry& geometry,
                DriverStreamHandle stream, void** kernel_params) override {
    ++calls;
    observed_function = function;
    observed_stream = stream;
    observed_grid_x = geometry.grid_x();
    observed_block_x = geometry.block_x();
    observed_params = kernel_params;
    return result;
  }

  int calls = 0;
  DriverFunctionHandle observed_function = 0;
  DriverStreamHandle observed_stream = 0;
  std::uint32_t observed_grid_x = 0;
  std::uint32_t observed_block_x = 0;
  void** observed_params = nullptr;
  Status result = Status::Ok();
};

KernelSignatureManifest scalar_signature() {
  const std::string digest(64, 'a');
  const std::vector<KernelParameterSpec> parameters{
      {"elements", KernelWireType::kU32, 0, "nonzero_elements_v1"}};
  return KernelSignatureManifest::Create("fill", digest, digest, parameters, 4)
      .value();
}

TEST(KernelLaunchGeometryTest, AcceptsBoundedThreeDimensionalLaunch) {
  auto geometry = KernelLaunchGeometry::Create(17, 2, 3, 32, 2, 1, 4096);
  ASSERT_TRUE(geometry.ok());
  EXPECT_EQ(geometry->grid_x(), 17);
  EXPECT_EQ(geometry->block_x(), 32);
  EXPECT_EQ(geometry->dynamic_shared_bytes(), 4096);
}

TEST(KernelLaunchGeometryTest, RejectsZeroAndPortableLimitViolations) {
  EXPECT_FALSE(KernelLaunchGeometry::Create(0, 1, 1, 32, 1, 1, 0).ok());
  EXPECT_FALSE(KernelLaunchGeometry::Create(1, 65536, 1, 32, 1, 1, 0).ok());
  EXPECT_FALSE(KernelLaunchGeometry::Create(1, 1, 1, 1024, 2, 1, 0).ok());
  EXPECT_FALSE(KernelLaunchGeometry::Create(1, 1, 1, 1, 1, 65, 0).ok());
}

TEST(VerifiedKernelLauncherTest, SubmitsReadyTypedPacketExactlyOnce) {
  auto packet = KernelArgumentPacket::Create(scalar_signature());
  ASSERT_TRUE(packet.ok());
  ASSERT_TRUE(packet->set_u32(0, 99).ok());
  auto geometry = KernelLaunchGeometry::Create(3, 1, 1, 64, 1, 1, 0);
  ASSERT_TRUE(geometry.ok());
  FakeLaunchDriver driver;
  const ResolvedKernelFunction function{"fill_u32", "fill",
                                        std::string(64, 'c'),
                                        std::string(64, 'a'), 23};

  const Status submitted =
      submit_verified_kernel(driver, function, geometry.value(), 41,
                             packet.value());
  ASSERT_TRUE(submitted.ok());
  EXPECT_EQ(driver.calls, 1);
  EXPECT_EQ(driver.observed_function, 23);
  EXPECT_EQ(driver.observed_stream, 41);
  EXPECT_EQ(driver.observed_grid_x, 3);
  EXPECT_EQ(driver.observed_block_x, 64);
  ASSERT_NE(driver.observed_params, nullptr);
  EXPECT_EQ(driver.observed_params[0], packet->argument_cell(0));
}

TEST(VerifiedKernelLauncherTest, RejectsBeforeDriverAndPropagatesFailure) {
  auto packet = KernelArgumentPacket::Create(scalar_signature());
  ASSERT_TRUE(packet.ok());
  auto geometry = KernelLaunchGeometry::Create(1, 1, 1, 32, 1, 1, 0);
  ASSERT_TRUE(geometry.ok());
  FakeLaunchDriver driver;
  const ResolvedKernelFunction function{"fill_u32", "fill",
                                        std::string(64, 'c'),
                                        std::string(64, 'a'), 23};
  EXPECT_FALSE(
      submit_verified_kernel(driver, function, geometry.value(), 41,
                             packet.value())
          .ok());
  EXPECT_EQ(driver.calls, 0);

  ASSERT_TRUE(packet->set_u32(0, 1).ok());
  const ResolvedKernelFunction null_function{"fill_u32", "fill",
                                             std::string(64, 'c'),
                                             std::string(64, 'a'), 0};
  EXPECT_FALSE(
      submit_verified_kernel(driver, null_function, geometry.value(), 41,
                             packet.value())
          .ok());
  EXPECT_FALSE(
      submit_verified_kernel(driver, function, geometry.value(), 0,
                             packet.value())
          .ok());
  const ResolvedKernelFunction wrong_abi{"fill_u32", "fill",
                                         std::string(64, 'c'),
                                         std::string(64, 'b'), 23};
  EXPECT_FALSE(submit_verified_kernel(driver, wrong_abi, geometry.value(), 41,
                                      packet.value())
                   .ok());
  EXPECT_EQ(driver.calls, 0);

  driver.result = Status::Internal("injected launch failure");
  const Status failed =
      submit_verified_kernel(driver, function, geometry.value(), 41,
                             packet.value());
  EXPECT_FALSE(failed.ok());
  EXPECT_EQ(driver.calls, 1);
}

}  // namespace
}  // namespace pih
