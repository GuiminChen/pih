#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include "pih/core/device.h"
#include "pih/core/dtype.h"

namespace pih {

class TensorView final {
 public:
  static constexpr std::size_t kMaxRank = 8;

  static Result<TensorView> Create(void* data, DType dtype,
                                   std::span<const std::int64_t> shape,
                                   std::span<const std::int64_t> strides,
                                   Device device,
                                   std::uint64_t generation = 0);

  [[nodiscard]] void* data() const noexcept { return data_; }
  [[nodiscard]] DType dtype() const noexcept { return dtype_; }
  [[nodiscard]] Device device() const noexcept { return device_; }
  [[nodiscard]] std::size_t rank() const noexcept { return rank_; }
  [[nodiscard]] std::uint64_t dim(std::size_t axis) const { return shape_.at(axis); }
  [[nodiscard]] std::uint64_t stride(std::size_t axis) const { return strides_.at(axis); }
  [[nodiscard]] std::uint64_t num_elements() const noexcept { return num_elements_; }
  [[nodiscard]] std::uint64_t byte_span() const noexcept { return byte_span_; }
  [[nodiscard]] std::uint64_t generation() const noexcept { return generation_; }

 private:
  TensorView(void* data, DType dtype, Device device, std::size_t rank,
             std::array<std::uint64_t, kMaxRank> shape,
             std::array<std::uint64_t, kMaxRank> strides,
             std::uint64_t num_elements, std::uint64_t byte_span,
             std::uint64_t generation)
      : data_(data),
        dtype_(dtype),
        device_(device),
        rank_(rank),
        shape_(shape),
        strides_(strides),
        num_elements_(num_elements),
        byte_span_(byte_span),
        generation_(generation) {}

  void* data_;
  DType dtype_;
  Device device_;
  std::size_t rank_;
  std::array<std::uint64_t, kMaxRank> shape_{};
  std::array<std::uint64_t, kMaxRank> strides_{};
  std::uint64_t num_elements_;
  std::uint64_t byte_span_;
  std::uint64_t generation_;
};

}  // namespace pih
