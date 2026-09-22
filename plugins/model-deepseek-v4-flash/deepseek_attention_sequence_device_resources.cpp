#include "pih/model/deepseek_attention_sequence_device_resources.h"

namespace pih {
namespace {

Status validate(const Buffer& buffer, std::uint64_t bytes,
                std::int32_t device_ordinal) {
  if (buffer.data() == nullptr || buffer.size_bytes() != bytes ||
      buffer.generation() == 0 ||
      buffer.device().type() != DeviceType::kCuda ||
      buffer.device().index() != device_ordinal) {
    return Status::FailedPrecondition(
        "DeepSeek attention state allocator returned invalid device storage");
  }
  return Status::Ok();
}

Result<Buffer> allocate(Allocator& allocator, std::uint64_t bytes,
                        std::int32_t device_ordinal) {
  auto result = Buffer::Allocate(allocator, bytes, 256);
  if (!result.ok()) return result.status();
  auto status = validate(*result, bytes, device_ordinal);
  if (!status.ok()) return status;
  return result;
}

}  // namespace

Result<DeepSeekAttentionSequenceDeviceResources>
DeepSeekAttentionSequenceDeviceResources::Allocate(
    Allocator& allocator, DeepSeekFixedStateLayout fixed_layout,
    std::uintptr_t completion_event,
    DeepSeekFixedStateBankOperations& fixed_operations,
    std::uint64_t context_identity, std::int32_t device_ordinal) {
  if (fixed_layout.total_bytes() == 0 || completion_event == 0 ||
      context_identity == 0 || device_ordinal < 0) {
    return Status::InvalidArgument(
        "DeepSeek attention sequence device identity is invalid");
  }
  auto fixed_first = allocate(
      allocator, fixed_layout.total_bytes(), device_ordinal);
  if (!fixed_first.ok()) return fixed_first.status();
  auto fixed_second = allocate(
      allocator, fixed_layout.total_bytes(), device_ordinal);
  if (!fixed_second.ok()) return fixed_second.status();
  auto fixed_banks = DeepSeekFixedStateBanks::Create(
      {reinterpret_cast<std::uintptr_t>(fixed_first->data()),
       fixed_first->size_bytes()},
      {reinterpret_cast<std::uintptr_t>(fixed_second->data()),
       fixed_second->size_bytes()}, completion_event, fixed_operations);
  if (!fixed_banks.ok()) return fixed_banks.status();
  return DeepSeekAttentionSequenceDeviceResources(
      std::move(*fixed_first), std::move(*fixed_second),
      std::move(fixed_layout), std::move(*fixed_banks));
}

}  // namespace pih
