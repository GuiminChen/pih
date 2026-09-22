#pragma once

#include <memory>
#include <vector>

#include "pih/model/deepseek_rank_plan_runtime.h"

namespace pih {

class DeepSeekRankBoundaryFactorySet final {
 public:
  static Result<DeepSeekRankBoundaryFactorySet> Create(
      std::uint32_t world_size,
      std::vector<std::unique_ptr<DeepSeekRankBoundaryFactory>> factories);

  DeepSeekRankBoundaryFactorySet(const DeepSeekRankBoundaryFactorySet&) =
      delete;
  DeepSeekRankBoundaryFactorySet& operator=(
      const DeepSeekRankBoundaryFactorySet&) = delete;
  DeepSeekRankBoundaryFactorySet(DeepSeekRankBoundaryFactorySet&&) noexcept =
      default;
  DeepSeekRankBoundaryFactorySet& operator=(
      DeepSeekRankBoundaryFactorySet&&) noexcept = default;

  [[nodiscard]] bool distributed() const noexcept { return world_size_ > 1; }
  [[nodiscard]] std::uint32_t world_size() const noexcept {
    return world_size_;
  }
  [[nodiscard]] DeepSeekRankBoundaryFactory* factory(
      std::uint32_t rank) noexcept;

 private:
  DeepSeekRankBoundaryFactorySet(
      std::uint32_t world_size,
      std::vector<std::unique_ptr<DeepSeekRankBoundaryFactory>> factories)
      noexcept
      : world_size_(world_size), factories_(std::move(factories)) {}

  std::uint32_t world_size_ = 0;
  std::vector<std::unique_ptr<DeepSeekRankBoundaryFactory>> factories_;
};

}  // namespace pih
