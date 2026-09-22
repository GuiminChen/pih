#include "pih/model/qwen3_int4_linear_shape_ledger.h"

#include <array>
#include <cstdint>
#include <map>
#include <set>
#include <string>

#include <gtest/gtest.h>

namespace pih {
namespace {

TEST(QwenInt4LinearShapeLedgerTest, FreezesOfficialNamesAndGeometry) {
  auto ledger = QwenInt4LinearShapeLedger::CreateOfficial();
  ASSERT_TRUE(ledger.ok()) << ledger.status().message();
  ASSERT_EQ(ledger->records().size(), 196U);

  std::set<std::string> source_names;
  std::map<QwenInt4LinearShapeFamily, std::uint64_t> family_counts;
  std::uint64_t packed_bytes = 0;
  std::uint64_t scale_bytes = 0;
  std::uint64_t bf16_bytes = 0;
  for (const auto& record : ledger->records()) {
    EXPECT_TRUE(source_names.insert(record.source_name).second);
    EXPECT_EQ(record.values_name, record.source_name + ".packed_values");
    EXPECT_EQ(record.scales_name, record.source_name + ".scales");
    EXPECT_EQ(record.columns % 128U, 0U);
    EXPECT_EQ(record.groups_per_row, record.columns / 128U);
    family_counts[record.family] += 1;
    packed_bytes += record.packed_bytes;
    scale_bytes += record.scale_bytes;
    bf16_bytes += record.bf16_bytes;
  }

  EXPECT_EQ(family_counts[QwenInt4LinearShapeFamily::kQProj], 28U);
  EXPECT_EQ(family_counts[QwenInt4LinearShapeFamily::kKvProj], 56U);
  EXPECT_EQ(family_counts[QwenInt4LinearShapeFamily::kOProj], 28U);
  EXPECT_EQ(family_counts[QwenInt4LinearShapeFamily::kGateUpProj], 56U);
  EXPECT_EQ(family_counts[QwenInt4LinearShapeFamily::kDownProj], 28U);
  EXPECT_EQ(packed_bytes, 220'200'960U);
  EXPECT_EQ(scale_bytes, 6'881'280U);
  EXPECT_EQ(bf16_bytes, 880'803'840U);
}

TEST(QwenInt4LinearShapeLedgerTest, UsesCanonicalLexicographicSourceOrder) {
  auto ledger = QwenInt4LinearShapeLedger::CreateOfficial();
  ASSERT_TRUE(ledger.ok());
  ASSERT_FALSE(ledger->records().empty());
  EXPECT_EQ(ledger->records().front().source_name,
            "model.layers.0.mlp.down_proj.weight");
  EXPECT_EQ(ledger->records().back().source_name,
            "model.layers.9.self_attn.v_proj.weight");
  for (std::size_t index = 1; index < ledger->records().size(); ++index) {
    EXPECT_LT(ledger->records()[index - 1].source_name,
              ledger->records()[index].source_name);
  }
}

TEST(QwenInt4LinearShapeLedgerTest, FreezesEveryFamilyRepresentative) {
  auto ledger = QwenInt4LinearShapeLedger::CreateOfficial();
  ASSERT_TRUE(ledger.ok());
  const std::array expected{
      std::tuple{QwenInt4LinearShapeFamily::kQProj, 2048ULL, 1024ULL,
                 1'048'576ULL, 32'768ULL},
      std::tuple{QwenInt4LinearShapeFamily::kKvProj, 1024ULL, 1024ULL,
                 524'288ULL, 16'384ULL},
      std::tuple{QwenInt4LinearShapeFamily::kOProj, 1024ULL, 2048ULL,
                 1'048'576ULL, 32'768ULL},
      std::tuple{QwenInt4LinearShapeFamily::kGateUpProj, 3072ULL, 1024ULL,
                 1'572'864ULL, 49'152ULL},
      std::tuple{QwenInt4LinearShapeFamily::kDownProj, 1024ULL, 3072ULL,
                 1'572'864ULL, 49'152ULL}};
  for (const auto& [family, rows, columns, packed, scales] : expected) {
    const auto* record = ledger->find_first(family);
    ASSERT_NE(record, nullptr);
    EXPECT_EQ(record->rows, rows);
    EXPECT_EQ(record->columns, columns);
    EXPECT_EQ(record->packed_bytes, packed);
    EXPECT_EQ(record->scale_bytes, scales);
  }
}

TEST(QwenInt4LinearShapeLedgerTest, HasStableSemanticIdentity) {
  auto ledger = QwenInt4LinearShapeLedger::CreateOfficial();
  ASSERT_TRUE(ledger.ok());
  auto digest = ledger->semantic_digest();
  ASSERT_TRUE(digest.ok());
  EXPECT_EQ(digest->hex(),
            "7556e8b3bc75a5de4b21ec9edafdeab0f97f46de9f723f111fcdf4d08a584b82");
}

}  // namespace
}  // namespace pih
