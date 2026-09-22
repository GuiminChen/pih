#pragma once

#include <cstdint>

#include "pih/core/buffer.h"

namespace pih {

class DeepSeekBoundarySendSource final {
 public:
  static constexpr std::uint64_t kWireBytesPerToken = 32768;

  static Result<DeepSeekBoundarySendSource> Create(
      Buffer& buffer, std::uint64_t offset_bytes,
      std::uint64_t wire_bytes, std::uint32_t token_count,
      std::uint64_t buffer_owner_id, std::uintptr_t context_identity,
      std::uint64_t producer_completion_generation);

  [[nodiscard]] void* data() const noexcept { return data_; }
  [[nodiscard]] Buffer& buffer() const noexcept { return *buffer_; }
  [[nodiscard]] std::uint64_t offset_bytes() const noexcept {
    return offset_bytes_;
  }
  [[nodiscard]] std::uint64_t wire_bytes() const noexcept {
    return wire_bytes_;
  }
  [[nodiscard]] std::uint32_t token_count() const noexcept {
    return token_count_;
  }
  [[nodiscard]] std::uint64_t buffer_owner_id() const noexcept {
    return buffer_owner_id_;
  }
  [[nodiscard]] std::uintptr_t context_identity() const noexcept {
    return context_identity_;
  }
  [[nodiscard]] std::uint64_t buffer_generation() const noexcept {
    return buffer_generation_;
  }
  [[nodiscard]] std::uint64_t producer_completion_generation() const noexcept {
    return producer_completion_generation_;
  }

 private:
  DeepSeekBoundarySendSource(
      Buffer& buffer, void* data, std::uint64_t offset_bytes,
      std::uint64_t wire_bytes, std::uint32_t token_count,
      std::uint64_t buffer_owner_id, std::uintptr_t context_identity,
      std::uint64_t producer_completion_generation) noexcept
      : buffer_(&buffer), data_(data), offset_bytes_(offset_bytes),
        wire_bytes_(wire_bytes), token_count_(token_count),
        buffer_owner_id_(buffer_owner_id),
        context_identity_(context_identity),
        buffer_generation_(buffer.generation()),
        producer_completion_generation_(producer_completion_generation) {}

  Buffer* buffer_;
  void* data_;
  std::uint64_t offset_bytes_;
  std::uint64_t wire_bytes_;
  std::uint32_t token_count_;
  std::uint64_t buffer_owner_id_;
  std::uintptr_t context_identity_;
  std::uint64_t buffer_generation_;
  std::uint64_t producer_completion_generation_;
};

}  // namespace pih
