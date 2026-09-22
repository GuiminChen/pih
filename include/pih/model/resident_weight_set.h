#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "pih/core/buffer.h"
#include "pih/core/memory_copier.h"
#include "pih/model/safetensors_file.h"

namespace pih {

struct WeightRequirement final {
  std::string name;
  DType dtype;
  std::vector<std::uint64_t> shape;
};

class ResidentWeightSet final {
 public:
  static Result<ResidentWeightSet> Load(
      const SafetensorsFile& source,
      std::span<const WeightRequirement> requirements, Allocator& allocator,
      MemoryCopier& copier, std::uint64_t alignment);

  ResidentWeightSet(const ResidentWeightSet&) = delete;
  ResidentWeightSet& operator=(const ResidentWeightSet&) = delete;
  ResidentWeightSet(ResidentWeightSet&&) noexcept = default;
  ResidentWeightSet& operator=(ResidentWeightSet&&) noexcept = default;

  [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }
  [[nodiscard]] std::uint64_t total_bytes() const noexcept { return total_bytes_; }
  Result<TensorView> tensor(std::string_view name) const;
  Result<std::uint64_t> generation(std::string_view name) const;

 private:
  struct Entry final {
    std::string name;
    DType dtype;
    std::vector<std::int64_t> shape;
    Buffer buffer;
  };

  ResidentWeightSet(std::vector<Entry> entries, std::uint64_t total_bytes)
      : entries_(std::move(entries)), total_bytes_(total_bytes) {}

  std::vector<Entry> entries_;
  std::uint64_t total_bytes_ = 0;
};

}  // namespace pih
