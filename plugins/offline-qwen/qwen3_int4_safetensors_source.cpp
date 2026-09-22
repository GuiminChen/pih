// Offline plugin-owned conversion implementation.
#include "qwen3_int4_safetensors_source.h"

#include <algorithm>

namespace pih {

Result<std::span<const std::byte>> verify_qwen_int4_bound_tensor(
    const ImmutableTensorBytes& tensor, const SafetensorRecord& physical,
    const QwenInt4ObservedSourceRecord& binding) {
  if (physical.name != binding.name || tensor.dtype != binding.dtype ||
      physical.dtype != binding.dtype ||
      !std::ranges::equal(tensor.shape, binding.shape) ||
      physical.shape != binding.shape ||
      physical.file_begin != binding.file_begin ||
      physical.file_end != binding.file_end ||
      tensor.bytes.size() != physical.file_end - physical.file_begin) {
    return Status::FailedPrecondition("Qwen INT4 bound source geometry changed");
  }
  auto digest = sha256(tensor.bytes);
  if (!digest.ok()) return digest.status();
  if (*digest != binding.payload_sha256) {
    return Status::FailedPrecondition("Qwen INT4 bound source payload changed");
  }
  return tensor.bytes;
}

Result<QwenInt4SafetensorsSource> QwenInt4SafetensorsSource::Create(
    const SafetensorsFile& source, const QwenInt4SourceBinding& binding) {
  if (source.header().tensors().size() != binding.records().size()) {
    return Status::FailedPrecondition("Qwen INT4 bound source count changed");
  }
  for (const auto& record : source.header().tensors()) {
    const auto* expected = binding.find(record.name);
    if (expected == nullptr || record.dtype != expected->dtype ||
        record.shape != expected->shape ||
        record.file_begin != expected->file_begin ||
        record.file_end != expected->file_end) {
      return Status::FailedPrecondition("Qwen INT4 bound source inventory changed");
    }
  }
  return QwenInt4SafetensorsSource(&source, &binding);
}

Result<std::span<const std::byte>> QwenInt4SafetensorsSource::tensor(
    std::string_view source_name) {
  const auto* expected = binding_->find(source_name);
  const auto* physical = source_->header().tensor(source_name);
  if (expected == nullptr || physical == nullptr) {
    return Status::InvalidArgument("Qwen INT4 bound source tensor is missing");
  }
  auto tensor = source_->tensor(source_name);
  if (!tensor.ok()) return tensor.status();
  return verify_qwen_int4_bound_tensor(*tensor, *physical, *expected);
}

}  // namespace pih
