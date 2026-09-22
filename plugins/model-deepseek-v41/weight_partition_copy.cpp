#include "weight_partition_copy.h"
#include "pih/core/checked_math.h"
#include <algorithm>
#include <new>

namespace pih::deepseek_v41 {
namespace {
Result<std::uint64_t> ElementBytes(WeightStorage storage) {
  switch (storage) {
    case WeightStorage::kBF16: return std::uint64_t{2};
    case WeightStorage::kF32: return std::uint64_t{4};
    case WeightStorage::kE4M3FN:
    case WeightStorage::kE8M0:
    case WeightStorage::kPackedE2M1: return std::uint64_t{1};
  }
  return Status::InvalidArgument("V4.1 copy storage type invalid");
}
Result<std::uint64_t> RowBytes(const RuntimeWeight& weight, std::uint64_t element) {
  if (weight.name.empty() || weight.name.size() > 256 || !weight.rows || !weight.columns ||
      (weight.dimensions != 1 && weight.dimensions != 2) ||
      (weight.dimensions == 1 && weight.columns != 1))
    return Status::InvalidArgument("V4.1 copy tensor geometry invalid");
  auto row = checked_mul_u64(weight.columns, element);
  if (!row.ok()) return row.status();
  auto bytes = checked_mul_u64(weight.rows, *row);
  if (!bytes.ok()) return bytes.status();
  if (*bytes != weight.bytes) return Status::InvalidArgument("V4.1 copy tensor byte count mismatch");
  return *row;
}
}

Result<WeightPartitionCopy> WeightPartitionCopy::Create(
    const RuntimeWeight& full, const WeightPartition& partition) {
  const auto& local = partition.tensor;
  if (full.name != local.name || full.storage != local.storage ||
      full.dimensions != local.dimensions || partition.axis < -1 || partition.axis > 1)
    return Status::InvalidArgument("V4.1 copy source/partition identity differs");
  auto element = ElementBytes(full.storage);
  if (!element.ok()) return element.status();
  auto source_row = RowBytes(full, *element);
  auto destination_row = RowBytes(local, *element);
  if (!source_row.ok()) return source_row.status();
  if (!destination_row.ok()) return destination_row.status();
  WeightPartitionCopy result;
  result.axis_ = partition.axis;
  result.source_bytes_ = full.bytes;
  result.destination_bytes_ = local.bytes;
  result.source_row_bytes_ = *source_row;
  result.destination_row_bytes_ = *destination_row;
  if (partition.axis == -1) {
    if (partition.first || partition.valid || partition.padding ||
        full.rows != local.rows || full.columns != local.columns)
      return Status::InvalidArgument("V4.1 replicated copy geometry differs");
    result.valid_bytes_ = local.bytes;
    return result;
  }
  const auto source_extent = partition.axis == 0 ? full.rows : full.columns;
  if (!partition.valid || partition.first > source_extent ||
      partition.valid > source_extent - partition.first)
    return Status::InvalidArgument("V4.1 copy partition is outside source extent");
  if (partition.axis == 0) {
    if (local.columns != full.columns || partition.valid > local.rows ||
        partition.padding != local.rows - partition.valid)
      return Status::InvalidArgument("V4.1 row copy or padding geometry differs");
    // Products fit because their factors are bounded by the validated full/local shapes.
    result.source_first_ = partition.first * *source_row;
    result.valid_bytes_ = partition.valid * *source_row;
    // E8M0 encodes numeric 1 as exponent byte 127. Engram scale padding
    // follows the reference's new_full(..., 1), not a raw-byte zero fill.
    if (partition.padding && full.storage == WeightStorage::kE8M0)
      result.padding_byte_ = std::byte{127};
  } else {
    if (partition.padding || local.rows != full.rows || local.columns != partition.valid || full.dimensions != 2)
      return Status::InvalidArgument("V4.1 column copy geometry differs");
    result.source_first_ = partition.first * *element;
    result.valid_bytes_ = local.bytes;
  }
  return result;
}

Result<WeightCopyRange> WeightPartitionCopy::Next(std::uint64_t offset, std::uint64_t maximum) const {
  if (offset >= destination_bytes_ || !maximum || maximum > 1024 * 1024)
    return Status::InvalidArgument("V4.1 copy cursor/workspace bound invalid");
  WeightCopyRange range{0, offset, std::min(maximum, destination_bytes_ - offset), false};
  if (axis_ == 1) {
    const auto row = offset / destination_row_bytes_;
    const auto column = offset % destination_row_bytes_;
    range.source_offset = row * source_row_bytes_ + source_first_ + column;
    range.bytes = std::min(range.bytes, destination_row_bytes_ - column);
  } else if (offset >= valid_bytes_) {
    range.padding = true;
    range.fill_byte = padding_byte_;
  } else {
    range.source_offset = source_first_ + offset;
    range.bytes = std::min(range.bytes, valid_bytes_ - offset);
  }
  if (!range.padding && (range.source_offset > source_bytes_ || range.bytes > source_bytes_ - range.source_offset))
    return Status::Internal("V4.1 copy cursor exceeded admitted source geometry");
  return range;
}

Status WeightPartitionCopy::Copy(const Reader& read, const Writer& write,
                                std::span<std::byte> workspace) const {
  if (!read || !write || workspace.empty() || workspace.size() > 1024 * 1024)
    return Status::InvalidArgument("V4.1 copy requires bounded workspace and exact I/O callbacks");
  try {
    for (std::uint64_t offset = 0; offset < destination_bytes_;) {
      auto range = Next(offset, workspace.size());
      if (!range.ok()) return range.status();
      auto chunk = workspace.first(static_cast<size_t>(range->bytes));
      if (range->padding) std::fill(chunk.begin(), chunk.end(), range->fill_byte);
      else {
        auto status = read(range->source_offset, chunk);
        if (!status.ok()) return status;
      }
      auto status = write(offset, chunk);
      if (!status.ok()) return status;
      offset += range->bytes;
    }
    return Status::Ok();
  } catch (...) {
    return Status::Internal("V4.1 partition copy callback failed; discard unpublished output");
  }
}
Status CopyCanonicalBackboneRank(const FlashConfig& config,
    std::uint32_t world, std::uint32_t rank,
    std::span<const RuntimeWeight> source_inventory,
    const CanonicalWeightReader& read, const PartitionWeightWriter& write,
    std::span<std::byte> workspace) {
  if (!read || !write || workspace.empty() || workspace.size() > 1024 * 1024)
    return Status::InvalidArgument("V4.1 rank copy requires bounded workspace and exact I/O callbacks");
  auto local = BackboneWeightInventory::Create(config, world, rank);
  if (!local.ok()) return local.status();
  auto full = BackboneWeightInventory::Create(config, 1, 0);
  if (!full.ok()) return full.status();
  auto admitted = full->Validate(source_inventory);
  if (!admitted.ok()) return admitted;
  try {
    // Admit every copy before the first write. This also prevents a late geometry
    // error from leaving an apparently successful prefix of rank tensors.
    std::vector<WeightPartitionCopy> copies;
    copies.reserve(local->weights().size());
    for (const auto& partition : local->weights()) {
      const auto all = full->weights();
      const auto source = std::lower_bound(all.begin(), all.end(), partition.tensor.name,
          [](const WeightPartition& item, const std::string& name) {
            return item.tensor.name < name;
          });
      if (source == all.end() || source->tensor.name != partition.tensor.name)
        return Status::Internal("V4.1 rank inventory member absent from full inventory");
      auto copy = WeightPartitionCopy::Create(source->tensor, partition);
      if (!copy.ok()) return copy.status();
      copies.push_back(std::move(*copy));
    }
    for (std::size_t index = 0; index < copies.size(); ++index) {
      const auto& tensor = local->weights()[index].tensor;
      auto status = copies[index].Copy(
          [&](std::uint64_t offset, std::span<std::byte> bytes) {
            return read(tensor.name, offset, bytes);
          },
          [&](std::uint64_t offset, std::span<const std::byte> bytes) {
            return write(tensor, offset, bytes);
          }, workspace);
      if (!status.ok()) return status;
    }
    return Status::Ok();
  } catch (const std::bad_alloc&) {
    return Status::ResourceExhausted("V4.1 rank copy allocation failed; discard unpublished output");
  } catch (...) {
    return Status::Internal("V4.1 rank copy failed; discard unpublished output");
  }
}
}  // namespace pih::deepseek_v41
