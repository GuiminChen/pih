#include "pih/model/deepseek_boundary_send_source.h"

#include <cstddef>

#include "pih/core/checked_math.h"

namespace pih {

Result<DeepSeekBoundarySendSource> DeepSeekBoundarySendSource::Create(
    Buffer& buffer, std::uint64_t offset_bytes,
    std::uint64_t wire_bytes, std::uint32_t token_count,
    std::uint64_t buffer_owner_id, std::uintptr_t context_identity,
    std::uint64_t producer_completion_generation) {
  if (token_count == 0 || buffer_owner_id == 0 || context_identity == 0 ||
      producer_completion_generation == 0 ||
      buffer.device().type() != DeviceType::kCuda ||
      buffer.generation() == 0 || buffer.data() == nullptr) {
    return Status::InvalidArgument(
        "DeepSeek boundary send source identity is invalid");
  }
  auto expected = checked_mul_u64(token_count, kWireBytesPerToken);
  auto end = checked_add_u64(offset_bytes, wire_bytes);
  if (!expected.ok()) return expected.status();
  if (!end.ok()) return end.status();
  if (wire_bytes != *expected || *end > buffer.size_bytes()) {
    return Status::InvalidArgument(
        "DeepSeek boundary send source extent is invalid");
  }
  auto* data = static_cast<std::byte*>(buffer.data()) + offset_bytes;
  return DeepSeekBoundarySendSource(
      buffer, data, offset_bytes, wire_bytes, token_count, buffer_owner_id,
      context_identity, producer_completion_generation);
}

}  // namespace pih
