#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "pih/core/device.h"
#include "pih/core/dtype.h"
#include "pih/core/tensor_view.h"

namespace pih {
namespace {

TEST(DTypeTest, ReportsClosedElementSizes) {
  EXPECT_EQ(dtype_size(DType::kFloat32).value(), 4);
  EXPECT_EQ(dtype_size(DType::kFloat16).value(), 2);
  EXPECT_EQ(dtype_size(DType::kBFloat16).value(), 2);
  EXPECT_EQ(dtype_size(DType::kFloat64).value(), 8);
  EXPECT_EQ(dtype_size(DType::kUInt32).value(), 4);
  EXPECT_EQ(dtype_size(DType::kInt8).value(), 1);
  EXPECT_EQ(dtype_size(static_cast<DType>(255)).status().code(),
            StatusCode::kInvalidArgument);
}

TEST(DeviceTest, ValidatesTypeAndIndex) {
  EXPECT_TRUE(Device::Create(DeviceType::kCpu, 0).ok());
  EXPECT_EQ(Device::Create(DeviceType::kCpu, 1).status().code(),
            StatusCode::kInvalidArgument);
  EXPECT_TRUE(Device::Create(DeviceType::kCuda, 3).ok());
  EXPECT_EQ(Device::Create(DeviceType::kCuda, -1).status().code(),
            StatusCode::kInvalidArgument);
  EXPECT_EQ(Device::Create(static_cast<DeviceType>(7), 0).status().code(),
            StatusCode::kInvalidArgument);
}

TEST(TensorViewTest, CreatesContiguousView) {
  std::array<std::uint16_t, 24> storage{};
  const std::array<std::int64_t, 3> shape{2, 3, 4};
  auto view = TensorView::Create(storage.data(), DType::kFloat16, shape, {},
                                 Device::Cpu());
  ASSERT_TRUE(view.ok());
  EXPECT_EQ(view->rank(), 3);
  EXPECT_EQ(view->num_elements(), 24);
  EXPECT_EQ(view->byte_span(), 48);
  EXPECT_EQ(view->stride(0), 12);
  EXPECT_EQ(view->stride(1), 4);
  EXPECT_EQ(view->stride(2), 1);
}

TEST(TensorViewTest, ScalarOccupiesOneElement) {
  float value = 0;
  auto view = TensorView::Create(&value, DType::kFloat32, {}, {}, Device::Cpu());
  ASSERT_TRUE(view.ok());
  EXPECT_EQ(view->rank(), 0);
  EXPECT_EQ(view->num_elements(), 1);
  EXPECT_EQ(view->byte_span(), 4);
}

TEST(TensorViewTest, ZeroExtentAllowsNullData) {
  const std::array<std::int64_t, 2> shape{0, 17};
  auto view = TensorView::Create(nullptr, DType::kBFloat16, shape, {}, Device::Cpu());
  ASSERT_TRUE(view.ok());
  EXPECT_EQ(view->num_elements(), 0);
  EXPECT_EQ(view->byte_span(), 0);
}

TEST(TensorViewTest, ExplicitStrideDeterminesAddressableSpan) {
  std::array<float, 16> storage{};
  const std::array<std::int64_t, 2> shape{2, 3};
  const std::array<std::int64_t, 2> strides{8, 2};
  auto view = TensorView::Create(storage.data(), DType::kFloat32, shape, strides,
                                 Device::Cpu());
  ASSERT_TRUE(view.ok());
  EXPECT_EQ(view->num_elements(), 6);
  EXPECT_EQ(view->byte_span(), 52);
}

TEST(TensorViewTest, RejectsInvalidMetadata) {
  std::byte storage[8]{};
  const std::array<std::int64_t, 1> negative_shape{-1};
  EXPECT_EQ(TensorView::Create(storage, DType::kInt8, negative_shape, {}, Device::Cpu())
                .status()
                .code(),
            StatusCode::kInvalidArgument);

  const std::array<std::int64_t, 2> shape{2, 2};
  const std::array<std::int64_t, 1> wrong_strides{1};
  EXPECT_EQ(TensorView::Create(storage, DType::kInt8, shape, wrong_strides, Device::Cpu())
                .status()
                .code(),
            StatusCode::kInvalidArgument);

  const std::array<std::int64_t, 2> negative_strides{2, -1};
  EXPECT_EQ(TensorView::Create(storage, DType::kInt8, shape, negative_strides,
                              Device::Cpu())
                .status()
                .code(),
            StatusCode::kInvalidArgument);

  const std::array<std::int64_t, 9> too_many_dims{1, 1, 1, 1, 1, 1, 1, 1, 1};
  EXPECT_EQ(TensorView::Create(storage, DType::kInt8, too_many_dims, {}, Device::Cpu())
                .status()
                .code(),
            StatusCode::kInvalidArgument);
}

TEST(TensorViewTest, RejectsNullForNonemptyAndSpanOverflow) {
  const std::array<std::int64_t, 1> one{1};
  EXPECT_EQ(TensorView::Create(nullptr, DType::kInt8, one, {}, Device::Cpu())
                .status()
                .code(),
            StatusCode::kInvalidArgument);

  std::byte storage[1]{};
  const std::array<std::int64_t, 2> shape{2, 2};
  const std::array<std::int64_t, 2> huge_stride{
      static_cast<std::int64_t>(std::numeric_limits<std::int64_t>::max()), 1};
  EXPECT_EQ(TensorView::Create(storage, DType::kFloat32, shape, huge_stride,
                              Device::Cpu())
                .status()
                .code(),
            StatusCode::kResourceExhausted);
}

}  // namespace
}  // namespace pih
