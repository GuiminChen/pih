#include "pih/model/safetensors_shard_index.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <limits>

#include "pih/core/bounded_json.h"

namespace pih { namespace {

bool valid_shard_name(std::string_view value) {
  if (value.empty() ||
      value.size() > SafetensorsShardIndex::kMaximumShardNameBytes ||
      !value.ends_with(".safetensors") || value.find("..") != value.npos) {
    return false;
  }
  for (const unsigned char byte : value) {
    if (!((byte >= 'a' && byte <= 'z') ||
          (byte >= 'A' && byte <= 'Z') ||
          (byte >= '0' && byte <= '9') || byte == '.' || byte == '_' ||
          byte == '-')) return false;
  }
  return true;
}

}  // namespace

Result<SafetensorsShardIndex> SafetensorsShardIndex::Parse(
    std::string_view json) {
  JsonLimits limits;
  limits.max_input_bytes = kMaximumIndexBytes;
  limits.max_depth = 8;
  limits.max_nodes = kMaximumTensorCount + 16;
  limits.max_string_bytes = kMaximumTensorNameBytes;
  auto root = JsonValue::Parse(json, limits);
  if (!root.ok()) return root.status();
  if (!root->is_object())
    return Status::InvalidArgument("Safetensors shard index must be an object");
  const auto* metadata = root->at("metadata");
  const auto* weights = root->at("weight_map");
  if (metadata == nullptr || !metadata->is_object() || weights == nullptr ||
      !weights->is_object() || weights->object().empty() ||
      weights->object().size() > kMaximumTensorCount) {
    return Status::InvalidArgument("Safetensors shard index structure is invalid");
  }
  const auto* total = metadata->at("total_size");
  if (total == nullptr || !total->is_integer() || total->integer() <= 0) {
    return Status::InvalidArgument("Safetensors shard total_size is invalid");
  }

  SafetensorsShardIndex result;
  result.total_size_ = static_cast<std::uint64_t>(total->integer());
  result.bindings_.reserve(weights->object().size());
  result.shard_names_.reserve(
      std::min(weights->object().size(), kMaximumShardCount));
  for (const auto& [tensor_name, shard] : weights->object()) {
    if (tensor_name.empty() || tensor_name.size() > kMaximumTensorNameBytes ||
        !shard.is_string() || !valid_shard_name(shard.string())) {
      return Status::InvalidArgument(
          "Safetensors shard binding is invalid");
    }
    result.bindings_.push_back({tensor_name, shard.string()});
    result.shard_names_.push_back(shard.string());
  }
  std::sort(result.bindings_.begin(), result.bindings_.end(),
            [](const auto& left, const auto& right) {
              return left.tensor_name < right.tensor_name;
            });
  std::sort(result.shard_names_.begin(), result.shard_names_.end());
  result.shard_names_.erase(
      std::unique(result.shard_names_.begin(), result.shard_names_.end()),
      result.shard_names_.end());
  if (result.shard_names_.size() > kMaximumShardCount) {
    return Status::ResourceExhausted(
        "Safetensors shard count exceeds object budget");
  }
  return result;
}

Status SafetensorsShardIndex::
validate_deepseek_flash_0731_repository_geometry() const {
  constexpr std::uint64_t kTotalSize = 166'878'536'440ULL;
  constexpr std::size_t kTensorCount = 72'317;
  constexpr std::size_t kShardCount = 48;
  if (total_size_ != kTotalSize || bindings_.size() != kTensorCount ||
      shard_names_.size() != kShardCount) {
    return Status::FailedPrecondition(
        "DeepSeek V4 Flash 0731 shard geometry is invalid");
  }
  for (std::size_t index = 0; index < kShardCount; ++index) {
    char expected[40]{};
    const auto written = std::snprintf(
        expected, sizeof(expected), "model-%05zu-of-00048.safetensors",
        index + 1);
    if (written <= 0 || static_cast<std::size_t>(written) >= sizeof(expected) ||
        shard_names_[index] != expected) {
      return Status::FailedPrecondition(
          "DeepSeek V4 Flash 0731 shard set is invalid");
    }
  }
  return Status::Ok();
}

Result<SafetensorsShardIndexFileReceipt> load_safetensors_shard_index_file(
    const std::filesystem::path& path) {
  if (path.empty())
    return Status::InvalidArgument("Safetensors shard index path is empty");
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream)
    return Status::Unavailable("cannot open Safetensors shard index");
  const auto end = stream.tellg();
  if (end <= 0 || static_cast<std::uint64_t>(end) >
                      SafetensorsShardIndex::kMaximumIndexBytes) {
    return Status::InvalidArgument(
        "Safetensors shard index file size is invalid");
  }
  std::string source(static_cast<std::size_t>(end), '\0');
  stream.seekg(0);
  if (!stream.read(source.data(), static_cast<std::streamsize>(source.size()))) {
    return Status::Unavailable("cannot read complete Safetensors shard index");
  }
  auto index = SafetensorsShardIndex::Parse(source);
  if (!index.ok()) return index.status();
  auto digest = sha256(std::as_bytes(std::span(source)));
  if (!digest.ok()) return digest.status();
  return SafetensorsShardIndexFileReceipt{
      std::move(*index), static_cast<std::uint64_t>(source.size()), *digest};
}

}  // namespace pih
