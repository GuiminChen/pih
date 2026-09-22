#include "pih/model/deepseek_mapped_tensor_source.h"

#include <algorithm>

#include "pih/core/checked_math.h"

namespace pih {

Result<DeepSeekMappedTensorSource> DeepSeekMappedTensorSource::Create(
    const DeepSeekStageMappedInventory& inventory,
    std::span<const DeepSeekRankTensorRecord> records) {
  if (records.empty()) {
    return Status::InvalidArgument("DeepSeek mapped tensor manifest is empty");
  }
  std::vector<DeepSeekRankTensorRecord> ordered(records.begin(), records.end());
  std::sort(ordered.begin(), ordered.end(), [](const auto& left, const auto& right) {
    return left.tensor_name < right.tensor_name;
  });
  for (std::size_t i = 0; i < ordered.size(); ++i) {
    const auto& record = ordered[i];
    if (record.tensor_name.empty() || record.shard_name.empty() ||
        record.file_begin >= record.file_end ||
        (i != 0 && ordered[i - 1].tensor_name == record.tensor_name)) {
      return Status::InvalidArgument("DeepSeek mapped tensor identity is invalid");
    }
    auto elements = std::uint64_t{1};
    for (const auto dimension : record.shape) {
      auto product = checked_mul_u64(elements, dimension);
      if (!product.ok()) return product.status();
      elements = *product;
    }
    auto element_bytes = dtype_size(record.dtype);
    if (!element_bytes.ok()) return element_bytes.status();
    auto logical_bytes = checked_mul_u64(elements, *element_bytes);
    if (!logical_bytes.ok()) return logical_bytes.status();
    if (*logical_bytes != record.file_end - record.file_begin) {
      return Status::InvalidArgument("DeepSeek mapped tensor byte geometry differs from dtype and shape");
    }
    auto mapped = inventory.bytes(record.shard_name, record.file_begin,
                                  *logical_bytes);
    if (!mapped.ok()) return mapped.status();
  }
  std::vector<const DeepSeekRankTensorRecord*> by_range;
  by_range.reserve(ordered.size());
  for (const auto& record : ordered) by_range.push_back(&record);
  std::sort(by_range.begin(), by_range.end(), [](const auto* left, const auto* right) {
    return left->shard_name < right->shard_name ||
           (left->shard_name == right->shard_name &&
            left->file_begin < right->file_begin);
  });
  for (std::size_t i = 1; i < by_range.size(); ++i) {
    if (by_range[i - 1]->shard_name == by_range[i]->shard_name &&
        by_range[i - 1]->file_end > by_range[i]->file_begin) {
      return Status::InvalidArgument("DeepSeek mapped tensor source ranges overlap");
    }
  }
  return DeepSeekMappedTensorSource(inventory, std::move(ordered));
}

Result<DeepSeekMappedTensorSpan> DeepSeekMappedTensorSource::resolve(
    std::string_view tensor_name) const {
  const auto found = std::lower_bound(
      records_.begin(), records_.end(), tensor_name,
      [](const auto& record, std::string_view name) {
        return record.tensor_name < name;
      });
  if (found == records_.end() || found->tensor_name != tensor_name) {
    return Status::InvalidArgument("DeepSeek mapped tensor does not exist");
  }
  auto bytes = inventory_->bytes(found->shard_name, found->file_begin,
                                 found->file_end - found->file_begin);
  if (!bytes.ok()) return bytes.status();
  return DeepSeekMappedTensorSpan{&*found, *bytes};
}

Result<std::span<const std::byte>>
DeepSeekMappedTensorSource::resolve_weight_bytes(
    std::string_view tensor_name) const {
  auto tensor = resolve(tensor_name);
  if (!tensor.ok()) return tensor.status();
  return tensor->bytes;
}

}  // namespace pih
