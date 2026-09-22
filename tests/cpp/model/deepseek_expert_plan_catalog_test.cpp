#include "pih/model/deepseek_expert_plan_catalog.h"

#include <gtest/gtest.h>

namespace pih { namespace {

DeepSeekPipelinePlanDescriptor descriptor() {
  return {7,13,DeepSeekPlanPhase::kDecode,1,1};
}

Result<DeepSeekExpertSubwavePlan> plan() {
  std::vector<DeepSeekExpertRoute> routes;
  for (std::uint16_t expert : {1,3,5,7,9,11}) {
    routes.push_back({expert,0,static_cast<std::uint8_t>(routes.size()),1.0F/6.0F});
  }
  return DeepSeekExpertSubwavePlan::Create(1,std::move(routes));
}

TEST(DeepSeekExpertPlanCatalogTest, PublishesOwnedLayerOnceForExactPlanIdentity) {
  auto catalog=DeepSeekExpertPlanCatalog::Create(descriptor(),{4,6});
  auto value=plan(); ASSERT_TRUE(catalog.ok()); ASSERT_TRUE(value.ok());
  ASSERT_TRUE(catalog->publish(5,std::move(*value)).ok());
  auto resolved=catalog->resolve(5,descriptor()); ASSERT_TRUE(resolved.ok());
  EXPECT_EQ((*resolved)->token_count(),1U);
  EXPECT_EQ(catalog->published_layers(),1U);
  auto duplicate=plan(); ASSERT_TRUE(duplicate.ok());
  EXPECT_FALSE(catalog->publish(5,std::move(*duplicate)).ok());
}

TEST(DeepSeekExpertPlanCatalogTest, RejectsMissingForeignAndStaleResolution) {
  auto catalog=DeepSeekExpertPlanCatalog::Create(descriptor(),{4,6});
  ASSERT_TRUE(catalog.ok());
  EXPECT_FALSE(catalog->resolve(4,descriptor()).ok());
  auto foreign=plan(); ASSERT_TRUE(foreign.ok());
  EXPECT_FALSE(catalog->publish(3,std::move(*foreign)).ok());
  auto value=plan(); ASSERT_TRUE(value.ok());
  ASSERT_TRUE(catalog->publish(4,std::move(*value)).ok());
  for (int field=0;field<5;++field) {
    auto stale=descriptor();
    if (field==0) ++stale.engine_epoch;
    if (field==1) ++stale.plan_sequence;
    if (field==2) stale.phase=DeepSeekPlanPhase::kVerify;
    if (field==3) ++stale.token_count;
    if (field==4) ++stale.sequence_count;
    EXPECT_FALSE(catalog->resolve(4,stale).ok());
  }
}

TEST(DeepSeekExpertPlanCatalogTest, RejectsTokenBoundMismatchAtPublication) {
  auto catalog=DeepSeekExpertPlanCatalog::Create(
      {7,13,DeepSeekPlanPhase::kDecode,2,2},{4,4});
  auto value=plan(); ASSERT_TRUE(catalog.ok()); ASSERT_TRUE(value.ok());
  EXPECT_FALSE(catalog->publish(4,std::move(*value)).ok());
}

TEST(DeepSeekExpertPlanCatalogTest, PublishesRouteSpanIntoReservedLayerBacking) {
  auto catalog=DeepSeekExpertPlanCatalog::Create(descriptor(),{4,4});
  auto value=plan(); ASSERT_TRUE(catalog.ok()); ASSERT_TRUE(value.ok());
  const auto routes=value->routes();
  ASSERT_TRUE(catalog->publish_routes(4,1,routes).ok());
  auto resolved=catalog->resolve(4,descriptor()); ASSERT_TRUE(resolved.ok());
  EXPECT_EQ((*resolved)->maximum_token_count(),1U);
  EXPECT_EQ((*resolved)->routes(),routes);
}

} }  // namespace pih
