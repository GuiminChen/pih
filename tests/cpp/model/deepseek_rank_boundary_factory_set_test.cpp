#include "pih/model/deepseek_rank_boundary_factory_set.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

class EmptyBoundaryFactory final : public DeepSeekRankBoundaryFactory {
 public:
  Result<DeepSeekPreparedRankBoundaries> prepare(
      DeepSeekPipelineTransaction&, const DeepSeekStagePlan&,
      std::uint32_t,
      std::uint32_t,
      const std::optional<DeepSeekBoundarySendSource>&) override {
    return Status::Internal("not used");
  }
};

TEST(DeepSeekRankBoundaryFactorySetTest, Pp1RequiresZeroFactories) {
  auto factories = DeepSeekRankBoundaryFactorySet::Create(1, {});
  ASSERT_TRUE(factories.ok()) << factories.status().message();
  EXPECT_FALSE(factories->distributed());
  EXPECT_EQ(factories->factory(0), nullptr);

  std::vector<std::unique_ptr<DeepSeekRankBoundaryFactory>> invalid;
  invalid.push_back(std::make_unique<EmptyBoundaryFactory>());
  EXPECT_FALSE(DeepSeekRankBoundaryFactorySet::Create(
      1, std::move(invalid)).ok());
}

TEST(DeepSeekRankBoundaryFactorySetTest, Pp2ThroughPp4RequireEveryRank) {
  for (std::uint32_t world_size = 2; world_size <= 4; ++world_size) {
    std::vector<std::unique_ptr<DeepSeekRankBoundaryFactory>> values;
    std::vector<DeepSeekRankBoundaryFactory*> identities;
    for (std::uint32_t rank = 0; rank < world_size; ++rank) {
      auto value = std::make_unique<EmptyBoundaryFactory>();
      identities.push_back(value.get());
      values.push_back(std::move(value));
    }
    auto factories = DeepSeekRankBoundaryFactorySet::Create(
        world_size, std::move(values));
    ASSERT_TRUE(factories.ok()) << factories.status().message();
    EXPECT_TRUE(factories->distributed());
    for (std::uint32_t rank = 0; rank < world_size; ++rank) {
      EXPECT_EQ(factories->factory(rank), identities[rank]);
    }
  }
}

TEST(DeepSeekRankBoundaryFactorySetTest, RejectsMissingOrNullRank) {
  std::vector<std::unique_ptr<DeepSeekRankBoundaryFactory>> missing;
  missing.push_back(std::make_unique<EmptyBoundaryFactory>());
  EXPECT_FALSE(DeepSeekRankBoundaryFactorySet::Create(
      2, std::move(missing)).ok());

  std::vector<std::unique_ptr<DeepSeekRankBoundaryFactory>> null_rank(2);
  null_rank[0] = std::make_unique<EmptyBoundaryFactory>();
  EXPECT_FALSE(DeepSeekRankBoundaryFactorySet::Create(
      2, std::move(null_rank)).ok());
}

}  // namespace
}  // namespace pih
