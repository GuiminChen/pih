#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "pih/core/result.h"
#include "pih/core/sha256.h"

namespace pih {

struct SafetensorsShardBinding final {
  std::string tensor_name;
  std::string shard_name;
  bool operator==(const SafetensorsShardBinding&) const = default;
};

class SafetensorsShardIndex final {
 public:
  static constexpr std::size_t kMaximumIndexBytes = 64ULL * 1024 * 1024;
  static constexpr std::size_t kMaximumTensorCount = 1'000'000;
  static constexpr std::size_t kMaximumTensorNameBytes = 1024;
  static constexpr std::size_t kMaximumShardCount = 1024;
  static constexpr std::size_t kMaximumShardNameBytes = 255;

  static Result<SafetensorsShardIndex> Parse(std::string_view json);
  [[nodiscard]] std::uint64_t total_size() const noexcept {
    return total_size_;
  }
  [[nodiscard]] const std::vector<SafetensorsShardBinding>& bindings()
      const noexcept { return bindings_; }
  [[nodiscard]] const std::vector<std::string>& shard_names() const noexcept {
    return shard_names_;
  }
  Status validate_deepseek_flash_0731_repository_geometry() const;

 private:
  std::uint64_t total_size_ = 0;
  std::vector<SafetensorsShardBinding> bindings_;
  std::vector<std::string> shard_names_;
};

struct SafetensorsShardIndexFileReceipt final {
  SafetensorsShardIndex index;
  std::uint64_t file_bytes = 0;
  Sha256Digest file_sha256;
};

Result<SafetensorsShardIndexFileReceipt> load_safetensors_shard_index_file(
    const std::filesystem::path& path);

}  // namespace pih
