#include "pih/model/deepseek_request_input_staging_resources.h"

#include <cstring>

#include "pih/core/checked_math.h"

namespace pih {

Result<DeepSeekRequestInputStagingResources>
DeepSeekRequestInputStagingResources::Allocate(
    std::uint32_t maximum_tokens, RegisteredPinnedAllocator& allocator) {
  if (maximum_tokens == 0 || maximum_tokens > 4096) {
    return Status::InvalidArgument(
        "DeepSeek request input staging capacity is invalid");
  }
  auto one = checked_mul_u64(maximum_tokens, sizeof(std::uint32_t));
  if (!one.ok()) return one.status();
  auto positions_offset = checked_align_up_u64(*one, 256);
  if (!positions_offset.ok()) return positions_offset.status();
  auto end = checked_add_u64(*positions_offset, *one);
  if (!end.ok()) return end.status();
  auto total = checked_align_up_u64(*end, 256);
  if (!total.ok()) return total.status();
  auto buffer = Buffer::Allocate(allocator, *total, 256);
  if (!buffer.ok()) return buffer.status();
  if (buffer->data() == nullptr || buffer->generation() == 0 ||
      buffer->device().type() != DeviceType::kCpu ||
      buffer->size_bytes() != *total) {
    return Status::FailedPrecondition(
        "DeepSeek request input staging allocator is invalid");
  }
  return DeepSeekRequestInputStagingResources(
      std::make_shared<Buffer>(std::move(*buffer)), *positions_offset,
      maximum_tokens);
}

Result<DeepSeekRequestInputStagingLease>
DeepSeekRequestInputStagingResources::stage(
    std::span<const std::uint32_t> token_ids,
    std::span<const std::uint32_t> positions,
    DeepSeekAttentionProjectionDeviceView destination,
    std::uintptr_t stream,
    DeepSeekRequestInputCopyOperations& operations) {
  if (pending_operations_ != nullptr || backing_ == nullptr || backing_.use_count() != 1 ||
      token_ids.empty() || token_ids.size() != positions.size() ||
      token_ids.size() > maximum_tokens_ ||
      destination.token_ids_u32 == 0 || destination.positions_u32 == 0 ||
      stream == 0) {
    return Status::ResourceExhausted(
        "DeepSeek request input staging is unavailable or invalid");
  }
  const auto bytes = token_ids.size_bytes();
  auto* base = static_cast<std::byte*>(backing_->data());
  std::memcpy(base, token_ids.data(), bytes);
  std::memcpy(base + positions_offset_, positions.data(), bytes);
  // Record ownership before the first call: failure or an exception may still
  // mean that the provider submitted a partial asynchronous copy.
  pending_operations_ = &operations;
  pending_stream_ = stream;
  auto status = operations.copy_h2d_async(
      destination.token_ids_u32, base, bytes, stream);
  if (status.ok()) {
    status = operations.copy_h2d_async(
        destination.positions_u32, base + positions_offset_, bytes, stream);
  }
  const auto retired = retire_pending();
  if (!retired.ok()) return retired;
  if (!status.ok()) return status;
  return DeepSeekRequestInputStagingLease{
      backing_, static_cast<std::uint32_t>(token_ids.size())};
}

Status DeepSeekRequestInputStagingResources::retire_pending() noexcept {
  if (pending_operations_ == nullptr) return Status::Ok();
  try {
    const auto status = pending_operations_->synchronize(pending_stream_);
    if (!status.ok()) return status;
    pending_operations_ = nullptr;
    pending_stream_ = 0;
    return Status::Ok();
  } catch (...) {
    return Status::Internal("DeepSeek request input synchronization failed");
  }
}

}  // namespace pih
