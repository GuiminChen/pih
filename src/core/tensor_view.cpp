#include "pih/core/tensor_view.h"

#include "pih/core/checked_math.h"

namespace pih {

Result<TensorView> TensorView::Create(void* data, DType dtype,
                                      std::span<const std::int64_t> shape,
                                      std::span<const std::int64_t> strides,
                                      Device device,
                                      std::uint64_t generation) {
  auto element_size = dtype_size(dtype);
  if (!element_size.ok()) {
    return element_size.status();
  }
  if (shape.size() > kMaxRank) {
    return Status::InvalidArgument("tensor rank exceeds eight");
  }
  if (!strides.empty() && strides.size() != shape.size()) {
    return Status::InvalidArgument("stride rank must equal shape rank");
  }

  std::array<std::uint64_t, kMaxRank> normalized_shape{};
  std::array<std::uint64_t, kMaxRank> normalized_strides{};
  std::uint64_t elements = 1;
  for (std::size_t axis = 0; axis < shape.size(); ++axis) {
    if (shape[axis] < 0) {
      return Status::InvalidArgument("tensor dimensions must be nonnegative");
    }
    normalized_shape[axis] = static_cast<std::uint64_t>(shape[axis]);
    auto product = checked_mul_u64(elements, normalized_shape[axis]);
    if (!product.ok()) {
      return product.status();
    }
    elements = product.value();
  }

  if (strides.empty()) {
    std::uint64_t next_stride = 1;
    for (std::size_t axis = shape.size(); axis > 0; --axis) {
      normalized_strides[axis - 1] = next_stride;
      auto product = checked_mul_u64(next_stride, normalized_shape[axis - 1]);
      if (!product.ok()) {
        return product.status();
      }
      next_stride = product.value();
    }
  } else {
    for (std::size_t axis = 0; axis < strides.size(); ++axis) {
      if (strides[axis] < 0) {
        return Status::InvalidArgument("tensor strides must be nonnegative");
      }
      normalized_strides[axis] = static_cast<std::uint64_t>(strides[axis]);
    }
  }

  std::uint64_t byte_span = 0;
  if (elements != 0) {
    std::uint64_t maximum_offset = 0;
    for (std::size_t axis = 0; axis < shape.size(); ++axis) {
      auto extent_offset = checked_mul_u64(normalized_shape[axis] - 1,
                                           normalized_strides[axis]);
      if (!extent_offset.ok()) {
        return extent_offset.status();
      }
      auto sum = checked_add_u64(maximum_offset, extent_offset.value());
      if (!sum.ok()) {
        return sum.status();
      }
      maximum_offset = sum.value();
    }
    auto addressed_elements = checked_add_u64(maximum_offset, 1);
    if (!addressed_elements.ok()) {
      return addressed_elements.status();
    }
    auto bytes = checked_mul_u64(addressed_elements.value(), element_size.value());
    if (!bytes.ok()) {
      return bytes.status();
    }
    byte_span = bytes.value();
    if (data == nullptr) {
      return Status::InvalidArgument("nonempty tensor requires data");
    }
  }

  return TensorView(data, dtype, device, shape.size(), normalized_shape,
                    normalized_strides, elements, byte_span, generation);
}

}  // namespace pih
