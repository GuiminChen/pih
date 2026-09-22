#include "pih/model/deepseek_learned_router_host_staging.h"

#include <limits>

namespace pih {
namespace {

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
  return (value + alignment - 1U) & ~(alignment - 1U);
}

}  // namespace

Result<DeepSeekLearnedRouterHostStaging>
DeepSeekLearnedRouterHostStaging::Allocate(
    std::uint32_t maximum_tokens, RegisteredPinnedAllocator& allocator) {
  if (maximum_tokens == 0 || maximum_tokens > 4096) {
    return Status::InvalidArgument(
        "DeepSeek learned router host staging capacity is invalid");
  }
  const auto score_count = static_cast<std::uint64_t>(maximum_tokens) *
      DeepSeekLearnedRouterCoordinator::kExpertCount;
  if (score_count > std::numeric_limits<std::uint64_t>::max() /
                        sizeof(float)) {
    return Status::ResourceExhausted(
        "DeepSeek learned router host staging size overflows");
  }
  const auto score_bytes = score_count * sizeof(float);
  const auto error_offset = align_up(score_bytes, kAlignment);
  const auto bytes = align_up(
      error_offset + sizeof(std::uint32_t), kAlignment);
  auto buffer = Buffer::Allocate(allocator, bytes, kAlignment);
  if (!buffer.ok()) return buffer.status();
  if (buffer->data() == nullptr || buffer->generation() == 0 ||
      buffer->device() != Device::Cpu() ||
      reinterpret_cast<std::uintptr_t>(buffer->data()) % kAlignment != 0) {
    return Status::FailedPrecondition(
        "DeepSeek pinned allocator returned invalid router staging");
  }
  return DeepSeekLearnedRouterHostStaging(
      std::move(*buffer), maximum_tokens, error_offset);
}

Result<std::span<float>> DeepSeekLearnedRouterHostStaging::scores(
    std::uint32_t token_count) noexcept {
  if (token_count == 0 || token_count > maximum_tokens_) {
    return Status::InvalidArgument(
        "DeepSeek learned router staging token count is invalid");
  }
  return std::span<float>(
      static_cast<float*>(buffer_.data()),
      static_cast<std::size_t>(token_count) *
          DeepSeekLearnedRouterCoordinator::kExpertCount);
}

std::uint32_t* DeepSeekLearnedRouterHostStaging::error_flag() noexcept {
  return reinterpret_cast<std::uint32_t*>(
      static_cast<std::byte*>(buffer_.data()) + error_offset_);
}

}  // namespace pih
