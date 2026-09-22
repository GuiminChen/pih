#pragma once

#include <cstdint>

#include "pih/core/result.h"

namespace pih {

class ElementwiseLaunch final {
 public:
  static constexpr std::uint32_t kWarpThreads = 32;
  static constexpr std::uint32_t kMaximumBlockThreads = 1024;
  static constexpr std::uint32_t kMaximumGridBlocks = 0x7fffffffU;

  static Result<ElementwiseLaunch> Create(std::uint64_t elements,
                                          std::uint32_t block_threads);

  [[nodiscard]] std::uint64_t elements() const noexcept { return elements_; }
  [[nodiscard]] std::uint32_t block_threads() const noexcept {
    return block_threads_;
  }
  [[nodiscard]] std::uint32_t grid_blocks() const noexcept {
    return grid_blocks_;
  }
  [[nodiscard]] std::uint64_t covered_threads() const noexcept {
    return static_cast<std::uint64_t>(block_threads_) * grid_blocks_;
  }

 private:
  ElementwiseLaunch(std::uint64_t elements, std::uint32_t block_threads,
                    std::uint32_t grid_blocks)
      : elements_(elements),
        block_threads_(block_threads),
        grid_blocks_(grid_blocks) {}

  std::uint64_t elements_;
  std::uint32_t block_threads_;
  std::uint32_t grid_blocks_;
};

}  // namespace pih
