#include "pih/model/deepseek_attention_projection_device_resources.h"

#include <array>

#include "pih/core/checked_math.h"

namespace pih {
namespace {

Result<std::uint64_t> reserve_rows(std::uint64_t& cursor,
                                   std::uint32_t maximum_tokens,
                                   std::uint64_t bytes_per_token) {
  auto bytes = checked_mul_u64(maximum_tokens, bytes_per_token);
  if (!bytes.ok()) return bytes.status();
  auto offset = checked_align_up_u64(cursor, 256);
  if (!offset.ok()) return offset.status();
  auto end = checked_add_u64(*offset, *bytes);
  if (!end.ok()) return end.status();
  cursor = *end;
  return *offset;
}

Result<std::uint64_t> reserve_bytes(std::uint64_t& cursor,
                                    std::uint64_t bytes) {
  auto offset = checked_align_up_u64(cursor, 256);
  if (!offset.ok()) return offset.status();
  auto end = checked_add_u64(*offset, bytes);
  if (!end.ok()) return end.status();
  cursor = *end;
  return *offset;
}

}  // namespace

Result<DeepSeekAttentionProjectionDeviceResources>
DeepSeekAttentionProjectionDeviceResources::Allocate(
    Allocator& allocator, std::uint32_t maximum_tokens,
    std::uint64_t context_identity, std::int32_t device_ordinal) {
  if (maximum_tokens == 0 || maximum_tokens > 4096 ||
      context_identity == 0 || device_ordinal < 0) {
    return Status::InvalidArgument(
        "DeepSeek attention projection device identity is invalid");
  }

  std::uint64_t cursor = 0;
  std::array<std::uint64_t, 18> offsets{};
  constexpr std::array<std::uint64_t, 17> bytes_per_token{
      4096,                         // input E4M3
      32,                           // input UE8M0 scales
      1024 * sizeof(std::uint16_t), // wq_a output
      1024 * sizeof(std::uint16_t), // q norm output
      1024,                         // q E4M3
      8,                            // q UE8M0 scales
      32768 * sizeof(std::uint16_t),// query
      512 * sizeof(std::uint16_t),  // kv
      32768 * sizeof(std::uint16_t),// sparse attention output
      32768,                        // grouped wo_a activation E4M3
      8 * 32,                       // grouped wo_a UE8M0 scales
      8192 * sizeof(std::uint16_t), // wo_a output
      8192,                         // wo_b activation E4M3
      64,                           // wo_b UE8M0 scales
      4096 * sizeof(std::uint16_t), // branch output
      sizeof(std::uint32_t),        // token ids
      sizeof(std::uint32_t),        // absolute positions
  };
  for (std::size_t index = 0; index < bytes_per_token.size(); ++index) {
    auto offset = reserve_rows(cursor, maximum_tokens,
                               bytes_per_token[index]);
    if (!offset.ok()) return offset.status();
    offsets[index] = *offset;
  }
  auto error_offset = reserve_bytes(cursor, sizeof(std::uint32_t));
  if (!error_offset.ok()) return error_offset.status();
  offsets.back() = *error_offset;
  auto total = checked_align_up_u64(cursor, 256);
  if (!total.ok()) return total.status();

  auto backing = Buffer::Allocate(allocator, *total, 256);
  if (!backing.ok()) return backing.status();
  if (backing->data() == nullptr || backing->size_bytes() != *total ||
      backing->generation() == 0 ||
      backing->device().type() != DeviceType::kCuda ||
      backing->device().index() != device_ordinal) {
    return Status::FailedPrecondition(
        "DeepSeek attention projection allocator returned invalid backing");
  }

  const auto base = reinterpret_cast<std::uintptr_t>(backing->data());
  const DeepSeekAttentionProjectionDeviceView view{
      base + offsets[0], base + offsets[1], base + offsets[2],
      base + offsets[3], base + offsets[4], base + offsets[5],
      base + offsets[6], base + offsets[7], base + offsets[8],
      base + offsets[9], base + offsets[10], base + offsets[11],
      base + offsets[12], base + offsets[13], base + offsets[14],
      base + offsets[15], base + offsets[16], base + offsets[17]};
  return DeepSeekAttentionProjectionDeviceResources(
      std::make_unique<Buffer>(std::move(*backing)), view, maximum_tokens,
      context_identity, device_ordinal);
}

}  // namespace pih
