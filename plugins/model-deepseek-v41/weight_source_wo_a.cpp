#include "weight_source_wo_a.h"
#include "config.h"
#include <limits>
#include <new>

namespace pih::deepseek_v41 {
Result<WoASourceBinding> WoASourceBinding::Create(std::uint32_t layer,
    const SafetensorsHeader& weight_header, std::string_view weight_name,
    const SafetensorsHeader& scale_header, std::string_view scale_name) {
  if (layer >= FlashConfig::kMainLayers || weight_name.size() > 256 || scale_name.size() > 256)
    return Status::InvalidArgument("V4.1 wo_a source layer/name bound invalid");
  try {
    // Restrict renaming to the exact reference prefix/attention aliases; no
    // substring replacement that could accidentally admit another operator.
    const auto layer_prefix = "layers." + std::to_string(layer) + ".";
    std::string prefix;
    for (const auto root : {"", "model."}) {
      for (const auto attention : {"self_attn.", "attn."}) {
        const auto candidate = std::string(root) + layer_prefix + attention + "wo_a.";
        if (weight_name == candidate + "weight") prefix = candidate;
      }
    }
    if (prefix.empty() || (scale_name != prefix + "weight_scale_inv" && scale_name != prefix + "scale"))
      return Status::InvalidArgument("V4.1 wo_a weight/scale must name the same source layer and operator");
    const auto* weight = weight_header.tensor(weight_name);
    const auto* scale = scale_header.tensor(scale_name);
    if (!weight || !scale || weight->dtype != DType::kFloat8E4M3 ||
        weight->shape.size() != 2 || weight->shape[0] != 8192 || weight->shape[1] != 4096 ||
        scale->shape.size() != 2 || !scale->shape[0] || !scale->shape[1])
      return Status::InvalidArgument("V4.1 wo_a source tensor absent or dtype/shape differs");
    if (8192 % scale->shape[0] || 4096 % scale->shape[1])
      return Status::InvalidArgument("V4.1 wo_a scale shape does not divide weight");
    const auto block = 8192 / scale->shape[0];
    if ((block != 32 && block != 128) || 4096 / scale->shape[1] != block)
      return Status::InvalidArgument("V4.1 wo_a scale must describe square 32/128 blocks");
    WoAScaleStorage storage;
    std::uint64_t scale_element = 0;
    switch (scale->dtype) {
      case DType::kFloat8E8M0: storage = WoAScaleStorage::kE8M0; scale_element = 1; break;
      case DType::kFloat32: storage = WoAScaleStorage::kF32LittleEndian; scale_element = 4; break;
      default: return Status::InvalidArgument("V4.1 wo_a scale must be E8M0 or F32");
    }
    const auto scale_bytes = scale->shape[0] * scale->shape[1] * scale_element;
    if (weight->file_end < weight->file_begin || scale->file_end < scale->file_begin ||
        weight->file_end - weight->file_begin != 8192ULL * 4096 ||
        scale->file_end - scale->file_begin != scale_bytes ||
        weight->file_end > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
        scale->file_end > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
      return Status::InvalidArgument("V4.1 wo_a source payload extent invalid");
    WoASourceBinding result;
    result.canonical_name_ = layer_prefix + "attn.wo_a.weight";
    result.geometry_ = {8192, 4096, scale->shape[0], scale->shape[1],
        8192ULL * 4096, scale_bytes, storage};
    result.weight_begin_ = weight->file_begin;
    result.scale_begin_ = scale->file_begin;
    return result;
  } catch (const std::bad_alloc&) {
    return Status::ResourceExhausted("V4.1 wo_a source binding allocation failed");
  }
}
Status WoASourceBinding::Convert(const WoATensorReader& weights,
    const WoATensorReader& scales, const WoATensorWriter& write) const {
  if (!weights || !scales || !write)
    return Status::InvalidArgument("V4.1 wo_a file conversion requires exact I/O callbacks");
  try {
    return DequantizeWoATensor(geometry_,
        [&](std::uint64_t offset, std::span<std::byte> bytes) {
          if (offset > geometry_.weight_bytes || bytes.size() > geometry_.weight_bytes - offset)
            return Status::Internal("V4.1 wo_a source read exceeds bound weight");
          return weights(weight_begin_ + offset, bytes);
        },
        [&](std::uint64_t offset, std::span<std::byte> bytes) {
          if (offset > geometry_.scale_bytes || bytes.size() > geometry_.scale_bytes - offset)
            return Status::Internal("V4.1 wo_a source read exceeds bound scale");
          return scales(scale_begin_ + offset, bytes);
        }, write);
  } catch (...) {
    return Status::Internal("V4.1 wo_a file conversion failed; discard unpublished output");
  }
}
}  // namespace pih::deepseek_v41
