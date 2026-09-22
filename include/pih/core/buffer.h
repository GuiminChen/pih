#pragma once

#include <cstdint>
#include <span>

#include "pih/core/allocator.h"
#include "pih/core/tensor_view.h"

namespace pih {

class Buffer final {
 public:
  static Result<Buffer> Allocate(Allocator& allocator, std::uint64_t bytes,
                                 std::uint64_t alignment);

  ~Buffer();
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  Buffer(Buffer&& other) noexcept;
  Buffer& operator=(Buffer&& other) noexcept;

  [[nodiscard]] void* data() const noexcept { return allocation_.data; }
  [[nodiscard]] std::uint64_t size_bytes() const noexcept { return allocation_.bytes; }
  [[nodiscard]] std::uint64_t generation() const noexcept {
    return allocation_.generation;
  }
  [[nodiscard]] Device device() const noexcept { return allocation_.device; }

  Result<TensorView> view(
      DType dtype, std::span<const std::int64_t> shape,
      std::span<const std::int64_t> strides = {}) const;

 private:
  Buffer(Allocator& allocator, Allocation allocation)
      : allocator_(&allocator), allocation_(allocation) {}
  void reset() noexcept;

  Allocator* allocator_ = nullptr;
  Allocation allocation_{};
};

}  // namespace pih
