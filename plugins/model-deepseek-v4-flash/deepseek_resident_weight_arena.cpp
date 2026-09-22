#include "pih/model/deepseek_resident_weight_arena.h"

#include <cstddef>
#include <limits>

namespace pih {

Result<DeepSeekResidentWeightArena> DeepSeekResidentWeightArena::Load(
    const DeepSeekWeightMaterializationPlan& plan,
    const DeepSeekWeightByteSource& source, Allocator& allocator,
    MemoryCopier& copier) {
  if (plan.copies().empty() || plan.backing_bytes() == 0 ||
      plan.payload_bytes() == 0) {
    return Status::InvalidArgument("DeepSeek resident weight plan is empty");
  }
  std::vector<std::span<const std::byte>> sources;
  sources.reserve(plan.copies().size());
  std::vector<Entry> entries;
  entries.reserve(plan.copies().size());
  for (const auto& copy : plan.copies()) {
    auto bytes = source.resolve_weight_bytes(copy.tensor_name);
    if (!bytes.ok()) return bytes.status();
    if (bytes->size() != copy.bytes ||
        copy.destination_offset > plan.backing_bytes() - copy.bytes) {
      return Status::InvalidArgument("DeepSeek weight source or destination span is invalid");
    }
    std::vector<std::int64_t> shape;
    shape.reserve(copy.shape.size());
    for (const auto dimension : copy.shape) {
      if (dimension > static_cast<std::uint64_t>(
                          std::numeric_limits<std::int64_t>::max())) {
        return Status::InvalidArgument("DeepSeek weight dimension exceeds tensor ABI");
      }
      shape.push_back(static_cast<std::int64_t>(dimension));
    }
    sources.push_back(*bytes);
    entries.push_back({copy.tensor_name, copy.dtype, std::move(shape),
                       copy.destination_offset});
  }
  auto backing = Buffer::Allocate(allocator, plan.backing_bytes(),
                                  DeepSeekWeightMaterializationPlan::kFinalAlignment);
  if (!backing.ok()) return backing.status();
  for (std::size_t i = 0; i < plan.copies().size(); ++i) {
    const auto& copy = plan.copies()[i];
    auto* destination = static_cast<std::byte*>(backing->data()) +
                        copy.destination_offset;
    auto status = copier.copy(destination, backing->device(), sources[i].data(),
                              Device::Cpu(), copy.bytes);
    if (!status.ok()) return status;
  }
  return DeepSeekResidentWeightArena(std::move(*backing), std::move(entries),
                                     plan.payload_bytes());
}

Result<TensorView> DeepSeekResidentWeightArena::tensor(
    std::string_view tensor_name) const {
  for (const auto& entry : entries_) {
    if (entry.name == tensor_name) {
      auto* data = static_cast<std::byte*>(backing_.data()) + entry.offset;
      return TensorView::Create(data, entry.dtype, entry.shape, {},
                                backing_.device(), backing_.generation());
    }
  }
  return Status::InvalidArgument("DeepSeek resident weight does not exist");
}

}  // namespace pih
