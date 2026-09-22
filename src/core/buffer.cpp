#include "pih/core/buffer.h"

#include <utility>

namespace pih {
namespace {

bool valid_alignment(std::uint64_t alignment) {
  return alignment != 0 && (alignment & (alignment - 1)) == 0;
}

}  // namespace

Result<Buffer> Buffer::Allocate(Allocator& allocator, std::uint64_t bytes,
                                std::uint64_t alignment) {
  if (!valid_alignment(alignment)) {
    return Status::InvalidArgument("alignment must be a nonzero power of two");
  }
  auto allocated = allocator.allocate(bytes, alignment);
  if (!allocated.ok()) {
    return allocated.status();
  }
  Allocation allocation = std::move(allocated).value();
  if (allocation.bytes != bytes || allocation.generation == 0 ||
      (bytes != 0 && allocation.data == nullptr)) {
    allocator.deallocate(allocation);
    return Status::Internal("allocator returned an invalid allocation");
  }
  return Buffer(allocator, allocation);
}

Buffer::~Buffer() { reset(); }

Buffer::Buffer(Buffer&& other) noexcept
    : allocator_(std::exchange(other.allocator_, nullptr)),
      allocation_(std::exchange(other.allocation_, Allocation{})) {}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
  if (this != &other) {
    reset();
    allocator_ = std::exchange(other.allocator_, nullptr);
    allocation_ = std::exchange(other.allocation_, Allocation{});
  }
  return *this;
}

Result<TensorView> Buffer::view(DType dtype,
                                std::span<const std::int64_t> shape,
                                std::span<const std::int64_t> strides) const {
  auto candidate = TensorView::Create(allocation_.data, dtype, shape, strides,
                                      allocation_.device, allocation_.generation);
  if (!candidate.ok()) {
    return candidate.status();
  }
  if (candidate->byte_span() > allocation_.bytes) {
    return Status::InvalidArgument("tensor view exceeds buffer allocation");
  }
  return candidate;
}

void Buffer::reset() noexcept {
  if (allocator_ != nullptr) {
    allocator_->deallocate(allocation_);
    allocator_ = nullptr;
    allocation_ = Allocation{};
  }
}

}  // namespace pih
