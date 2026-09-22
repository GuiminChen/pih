#include "pih/model/deepseek_rank_boundary_factory_set.h"

#include <algorithm>
#include <utility>

namespace pih {

Result<DeepSeekRankBoundaryFactorySet>
DeepSeekRankBoundaryFactorySet::Create(
    std::uint32_t world_size,
    std::vector<std::unique_ptr<DeepSeekRankBoundaryFactory>> factories) {
  if (world_size == 0 || world_size > 4) {
    return Status::InvalidArgument(
        "DeepSeek boundary factory world size is invalid");
  }
  const std::size_t required = world_size == 1 ? 0 : world_size;
  if (factories.size() != required ||
      std::ranges::any_of(factories,
                          [](const auto& value) { return value == nullptr; })) {
    return Status::InvalidArgument(
        "DeepSeek boundary factory set differs from rank topology");
  }
  return DeepSeekRankBoundaryFactorySet(world_size, std::move(factories));
}

DeepSeekRankBoundaryFactory* DeepSeekRankBoundaryFactorySet::factory(
    std::uint32_t rank) noexcept {
  if (rank >= world_size_ || factories_.empty()) return nullptr;
  return factories_[rank].get();
}

}  // namespace pih
