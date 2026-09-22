#include "pih/model/resident_weight_set.h"

#include <limits>
#include <unordered_set>
#include <utility>

#include "pih/core/checked_math.h"

namespace pih {
namespace {

struct ValidatedWeight final {
  const WeightRequirement* requirement;
  ImmutableTensorBytes source;
  std::vector<std::int64_t> shape;
};

}  // namespace

Result<ResidentWeightSet> ResidentWeightSet::Load(
    const SafetensorsFile& source,
    std::span<const WeightRequirement> requirements, Allocator& allocator,
    MemoryCopier& copier, std::uint64_t alignment) {
  std::vector<ValidatedWeight> validated;
  validated.reserve(requirements.size());
  std::unordered_set<std::string_view> names;
  names.reserve(requirements.size());
  std::uint64_t total_bytes = 0;

  // Validate the complete manifest before allocating so malformed input cannot
  // expose a partial resident catalog or consume device memory.
  for (const auto& requirement : requirements) {
    if (requirement.name.empty()) {
      return Status::InvalidArgument("weight requirement name must not be empty");
    }
    if (!names.insert(requirement.name).second) {
      return Status::InvalidArgument("weight requirements contain a duplicate name");
    }
    auto tensor = source.tensor(requirement.name);
    if (!tensor.ok()) return tensor.status();
    if (tensor->dtype != requirement.dtype || tensor->shape.size() != requirement.shape.size()) {
      return Status::InvalidArgument("weight requirement does not match source tensor");
    }
    std::vector<std::int64_t> shape;
    shape.reserve(requirement.shape.size());
    for (std::size_t axis = 0; axis < requirement.shape.size(); ++axis) {
      if (tensor->shape[axis] != requirement.shape[axis] ||
          requirement.shape[axis] > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
        return Status::InvalidArgument("weight requirement does not match source tensor");
      }
      shape.push_back(static_cast<std::int64_t>(requirement.shape[axis]));
    }
    auto next_total = checked_add_u64(total_bytes, tensor->bytes.size());
    if (!next_total.ok()) return next_total.status();
    total_bytes = next_total.value();
    validated.push_back(ValidatedWeight{&requirement, tensor.value(), std::move(shape)});
  }

  std::vector<Entry> entries;
  entries.reserve(validated.size());
  for (auto& item : validated) {
    auto buffer = Buffer::Allocate(allocator, item.source.bytes.size(), alignment);
    if (!buffer.ok()) return buffer.status();
    const Status copied = copier.copy(buffer->data(), buffer->device(),
                                      item.source.bytes.data(), Device::Cpu(),
                                      item.source.bytes.size());
    if (!copied.ok()) return copied;
    entries.push_back(Entry{item.requirement->name, item.requirement->dtype,
                            std::move(item.shape), std::move(buffer).value()});
  }
  return ResidentWeightSet(std::move(entries), total_bytes);
}

Result<TensorView> ResidentWeightSet::tensor(std::string_view name) const {
  for (const auto& entry : entries_) {
    if (entry.name == name) return entry.buffer.view(entry.dtype, entry.shape);
  }
  return Status::InvalidArgument("resident weight does not exist");
}

Result<std::uint64_t> ResidentWeightSet::generation(std::string_view name) const {
  for (const auto& entry : entries_) {
    if (entry.name == name) return entry.buffer.generation();
  }
  return Status::InvalidArgument("resident weight does not exist");
}

}  // namespace pih
