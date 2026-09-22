// Offline plugin-owned conversion implementation.
#include "qwen3_int4_tensor_payload.h"

#include <limits>

#include "pih/core/bfloat16.h"
#include "pih/core/checked_math.h"
#include "qwen3_int4_reference_quantizer.h"

namespace pih {

Result<QwenInt4TensorPayload> QwenInt4TensorPayload::Create(
    std::span<const std::byte> source_bf16_le,
    const QwenInt4LinearRecordPlan& plan) {
  if (source_bf16_le.size() != plan.bf16_bytes ||
      (source_bf16_le.size() & 1U) != 0 ||
      plan.columns % QwenInt4QuantizedTensor::kGroupSize != 0) {
    return Status::InvalidArgument("Qwen INT4 source Linear geometry mismatch");
  }
  const std::size_t elements = source_bf16_le.size() / 2;
  std::vector<BFloat16> decoded(elements);
  for (std::size_t index = 0; index < elements; ++index) {
    decoded[index].bits =
        std::to_integer<std::uint16_t>(source_bf16_le[index * 2]) |
        (std::to_integer<std::uint16_t>(source_bf16_le[index * 2 + 1]) << 8U);
  }
  auto quantized = QwenInt4QuantizedTensor::Quantize(
      decoded, plan.rows, plan.columns);
  if (!quantized.ok()) return quantized.status();
  const auto logical_bytes = quantized->logical_values().size();
  const auto packed_bytes = quantized->packed_values().size();
  const auto scale_bytes = quantized->scale_bits().size() * 2;
  auto peak = checked_add_u64(decoded.size() * sizeof(BFloat16), logical_bytes);
  if (!peak.ok()) return peak.status();
  peak = checked_add_u64(*peak, packed_bytes);
  if (!peak.ok()) return peak.status();
  peak = checked_add_u64(*peak, scale_bytes);
  if (!peak.ok()) return peak.status();

  auto payload = std::move(*quantized).release_artifact_payload();
  std::vector<std::byte> scales(payload.scale_bits.size() * 2);
  for (std::size_t index = 0; index < payload.scale_bits.size(); ++index) {
    scales[index * 2] = static_cast<std::byte>(payload.scale_bits[index]);
    scales[index * 2 + 1] =
        static_cast<std::byte>(payload.scale_bits[index] >> 8U);
  }
  if (payload.packed_values.size() != plan.packed_bytes ||
      scales.size() != plan.scale_bytes) {
    return Status::Internal("Qwen INT4 quantizer payload geometry drifted");
  }
  auto packed_digest = sha256(payload.packed_values);
  if (!packed_digest.ok()) return packed_digest.status();
  auto scale_digest = sha256(scales);
  if (!scale_digest.ok()) return scale_digest.status();
  return QwenInt4TensorPayload(std::move(payload.packed_values),
                               std::move(scales), *packed_digest,
                               *scale_digest, *peak);
}

}  // namespace pih
