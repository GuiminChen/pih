#include "pih/model/qwen3_int4_disposition_plan.h"

#include <map>
#include <set>
#include <string>

#include <gtest/gtest.h>

namespace pih {
namespace {

TEST(QwenInt4DispositionPlanTest, ConsumesEveryOfficialSourceExactlyOnce) {
  auto plan = QwenInt4DispositionPlan::CreateOfficialPureW4();
  ASSERT_TRUE(plan.ok()) << plan.status().message();
  ASSERT_EQ(plan->records().size(), 311U);
  std::set<std::string> sources;
  std::map<QwenInt4DispositionKind, std::size_t> counts;
  for (const auto& record : plan->records()) {
    EXPECT_TRUE(sources.insert(record.source_name).second);
    ++counts[record.kind];
  }
  EXPECT_EQ(counts[QwenInt4DispositionKind::kQuantizeW4A16G128], 196U);
  EXPECT_EQ(counts[QwenInt4DispositionKind::kIdentityCopyBf16], 114U);
  EXPECT_EQ(counts[QwenInt4DispositionKind::kEqualSourceDedupToAlias], 1U);
}

TEST(QwenInt4DispositionPlanTest, DerivesEveryTargetExactlyOnce) {
  auto plan = QwenInt4DispositionPlan::CreateOfficialPureW4();
  ASSERT_TRUE(plan.ok());
  std::set<std::string> targets;
  for (const auto& record : plan->records()) {
    for (const auto& target : record.target_identities) {
      EXPECT_TRUE(targets.insert(target).second);
    }
  }
  EXPECT_EQ(targets.size(), 507U);
  EXPECT_TRUE(targets.contains("model.embed_tokens.weight"));
  EXPECT_TRUE(targets.contains("lm_head.weight"));
}

TEST(QwenInt4DispositionPlanTest, PinsPhysicalTiedSourcesBeforeTargetAlias) {
  auto plan = QwenInt4DispositionPlan::CreateOfficialPureW4();
  ASSERT_TRUE(plan.ok());
  const auto* owner = plan->find("model.embed_tokens.weight");
  const auto* alias = plan->find("lm_head.weight");
  ASSERT_NE(owner, nullptr);
  ASSERT_NE(alias, nullptr);
  EXPECT_EQ(owner->kind, QwenInt4DispositionKind::kIdentityCopyBf16);
  EXPECT_EQ(alias->kind, QwenInt4DispositionKind::kEqualSourceDedupToAlias);
  EXPECT_EQ(owner->source_bytes, 311'164'928U);
  EXPECT_EQ(alias->source_bytes, 311'164'928U);
  EXPECT_EQ(alias->equality_owner_source, "model.embed_tokens.weight");
  ASSERT_EQ(alias->target_identities.size(), 1U);
  EXPECT_EQ(alias->target_identities.front(), "lm_head.weight");
}

TEST(QwenInt4DispositionPlanTest, FreezesStableDispositionIdentity) {
  auto plan = QwenInt4DispositionPlan::CreateOfficialPureW4();
  ASSERT_TRUE(plan.ok());
  auto digest = plan->semantic_digest();
  ASSERT_TRUE(digest.ok());
  EXPECT_EQ(digest->hex(),
            "df673ffd92640d34675c2c87e2d37ee5c84952ce249016089208bc921c91ee0c");
}

}  // namespace
}  // namespace pih
