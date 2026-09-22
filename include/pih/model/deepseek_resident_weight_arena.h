#pragma once

#include <string_view>
#include <vector>

#include "pih/core/buffer.h"
#include "pih/core/memory_copier.h"
#include "pih/model/deepseek_mapped_tensor_source.h"
#include "pih/model/deepseek_weight_materialization_plan.h"

namespace pih {

class DeepSeekResidentWeightArena final {
 public:
  static Result<DeepSeekResidentWeightArena> Load(
      const DeepSeekWeightMaterializationPlan& plan,
      const DeepSeekWeightByteSource& source, Allocator& allocator,
      MemoryCopier& copier);

  DeepSeekResidentWeightArena(const DeepSeekResidentWeightArena&) = delete;
  DeepSeekResidentWeightArena& operator=(const DeepSeekResidentWeightArena&) = delete;
  DeepSeekResidentWeightArena(DeepSeekResidentWeightArena&&) noexcept = default;
  DeepSeekResidentWeightArena& operator=(DeepSeekResidentWeightArena&&) noexcept = default;

  Result<TensorView> tensor(std::string_view tensor_name) const;
  [[nodiscard]] std::uint64_t generation() const noexcept {
    return backing_.generation();
  }
  [[nodiscard]] std::uint64_t backing_bytes() const noexcept {
    return backing_.size_bytes();
  }
  [[nodiscard]] std::uint64_t payload_bytes() const noexcept {
    return payload_bytes_;
  }
  [[nodiscard]] std::uintptr_t base_address() const noexcept {
    return reinterpret_cast<std::uintptr_t>(backing_.data());
  }
  [[nodiscard]] Device device() const noexcept { return backing_.device(); }

 private:
  struct Entry final {
    std::string name;
    DType dtype{};
    std::vector<std::int64_t> shape;
    std::uint64_t offset = 0;
  };

  DeepSeekResidentWeightArena(Buffer backing, std::vector<Entry> entries,
                              std::uint64_t payload_bytes)
      : backing_(std::move(backing)), entries_(std::move(entries)),
        payload_bytes_(payload_bytes) {}

  Buffer backing_;
  std::vector<Entry> entries_;
  std::uint64_t payload_bytes_ = 0;
};

}  // namespace pih
