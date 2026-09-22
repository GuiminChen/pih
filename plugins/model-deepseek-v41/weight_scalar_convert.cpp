#include "weight_scalar_convert.h"
#include "pih/core/bfloat16.h"
#include "pih/core/checked_math.h"
#include <algorithm>
#include <bit>
#include <new>

namespace pih::deepseek_v41 {
Status ValidateScalarWeight(const RuntimeWeight& target, const SafetensorRecord& source) {
  if ((target.dimensions != 1 && target.dimensions != 2) ||
      source.shape.size() != target.dimensions || !target.rows || !target.columns ||
      (target.dimensions == 1 && target.columns != 1) || source.shape[0] != target.rows ||
      (target.dimensions == 2 && source.shape[1] != target.columns))
    return Status::InvalidArgument("V4.1 scalar conversion shape/callback invalid");
  const bool source_bf16 = source.dtype == DType::kBFloat16;
  const bool source_f32 = source.dtype == DType::kFloat32;
  bool allowed = false;
  std::uint64_t output_element = 0;
  switch (target.storage) {
    case WeightStorage::kBF16: allowed = source_bf16 || source_f32; output_element = 2; break;
    case WeightStorage::kF32: allowed = source_bf16 || source_f32; output_element = 4; break;
    case WeightStorage::kE4M3FN: allowed = source.dtype == DType::kFloat8E4M3; output_element = 1; break;
    case WeightStorage::kE8M0: allowed = source.dtype == DType::kFloat8E8M0 || source_f32; output_element = 1; break;
    case WeightStorage::kPackedE2M1: allowed = source.dtype == DType::kInt8 || source.dtype == DType::kUInt8; output_element = 1; break;
  }
  if (!allowed) return Status::InvalidArgument("V4.1 scalar source/target dtype conversion unsupported");
  auto input_element = dtype_size(source.dtype); if (!input_element.ok()) return input_element.status();
  auto elements = checked_mul_u64(target.rows, target.columns); if (!elements.ok()) return elements.status();
  auto input_bytes = checked_mul_u64(*elements, *input_element); if (!input_bytes.ok()) return input_bytes.status();
  auto output_bytes = checked_mul_u64(*elements, output_element); if (!output_bytes.ok()) return output_bytes.status();
  if (source.file_end < source.file_begin || source.file_end - source.file_begin != *input_bytes || target.bytes != *output_bytes)
    return Status::InvalidArgument("V4.1 scalar conversion byte extents differ");
  return Status::Ok();
}
Status ConvertScalarWeight(const RuntimeWeight& target, const SafetensorRecord& source,
    const WoATensorReader& read, const WoATensorWriter& write) {
  if (!read || !write) return Status::InvalidArgument("V4.1 scalar conversion callbacks absent");
  auto admitted = ValidateScalarWeight(target, source);
  if (!admitted.ok()) return admitted;
  const bool source_bf16 = source.dtype == DType::kBFloat16;
  const bool source_f32 = source.dtype == DType::kFloat32;
  const std::uint64_t output_element = target.storage == WeightStorage::kBF16 ? 2 :
      target.storage == WeightStorage::kF32 ? 4 : 1;
  auto input_element = dtype_size(source.dtype);
  auto elements = checked_mul_u64(target.rows, target.columns);
  if (!input_element.ok()) return input_element.status();
  if (!elements.ok()) return elements.status();
  try {
    const auto batch = std::min<std::uint64_t>(*elements, (1024 * 1024) / std::max(*input_element, output_element));
    std::vector<std::byte> input(static_cast<std::size_t>(batch * *input_element));
    std::vector<std::byte> output(static_cast<std::size_t>(batch * output_element));
    for (std::uint64_t first = 0; first < *elements;) {
      const auto count = std::min(batch, *elements - first);
      auto in = std::span(input).first(static_cast<std::size_t>(count * *input_element));
      auto out = std::span(output).first(static_cast<std::size_t>(count * output_element));
      auto status = read(first * *input_element, in); if (!status.ok()) return status;
      for (std::uint64_t i = 0; i < count; ++i) {
        std::uint32_t bits = 0;
        for (unsigned byte = 0; byte < *input_element; ++byte)
          bits |= std::to_integer<std::uint32_t>(in[i * *input_element + byte]) << (8 * byte);
        if (source_bf16) bits <<= 16;
        if ((source_bf16 || source_f32) && (bits & 0x7f800000U) == 0x7f800000U)
          return Status::InvalidArgument("V4.1 scalar source contains nonfinite value");
        switch (target.storage) {
          case WeightStorage::kBF16:
            bits = BFloat16::FromFloat(std::bit_cast<float>(bits)).bits;
            if ((bits & 0x7f80U) == 0x7f80U)
              return Status::InvalidArgument("V4.1 scalar BF16 conversion overflow");
            break;
          case WeightStorage::kF32: break;
          case WeightStorage::kE8M0:
            if (source_f32) {
              if (bits == 0x00400000U) bits = 0; // exact 2^-127
              else if (!(bits & 0x807fffffU) && (bits >> 23) > 0 && (bits >> 23) < 255)
                bits >>= 23;
              else return Status::InvalidArgument("V4.1 scale is not an exactly representable positive E8M0 power");
            } else if (bits == 255) return Status::InvalidArgument("V4.1 E8M0 scale is NaN");
            break;
          case WeightStorage::kE4M3FN:
            if ((bits & 127U) == 127U) return Status::InvalidArgument("V4.1 E4M3FN weight is NaN");
            break;
          case WeightStorage::kPackedE2M1: break; // preserve both packed nibbles, not numeric int8 cast
        }
        for (unsigned byte = 0; byte < output_element; ++byte)
          out[i * output_element + byte] = static_cast<std::byte>((bits >> (8 * byte)) & 255U);
      }
      status = write(first * output_element, out); if (!status.ok()) return status;
      first += count;
    }
    return Status::Ok();
  } catch (const std::bad_alloc&) {
    return Status::ResourceExhausted("V4.1 scalar conversion allocation failed; discard unpublished output");
  } catch (...) {
    return Status::Internal("V4.1 scalar conversion failed; discard unpublished output");
  }
}
}  // namespace pih::deepseek_v41
