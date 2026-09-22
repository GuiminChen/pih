#include "pih/core/canonical_hash.h"

#include <gtest/gtest.h>

namespace pih { namespace {
TEST(CanonicalHashTest, MatchesNormativeZeroFieldVector) {
  auto hash = CanonicalHashBuilder::Create("pih:hash-golden:v1", 0);
  ASSERT_TRUE(hash.ok());
  EXPECT_EQ(hash->finalize().value().hex(),
            "6c975ea115b57de4d4ddea31296c92d49bb2c0a4dde8a31645b44191034928e7");
}
TEST(CanonicalHashTest, MatchesNormativeU32Vector) {
  auto hash = CanonicalHashBuilder::Create("pih:hash-golden:v1", 1);
  ASSERT_TRUE(hash.ok()); ASSERT_TRUE(hash->add_u32(1, 1).ok());
  EXPECT_EQ(hash->finalize().value().hex(),
            "b11676d18ceada23b9288fdccf82181cf7f2773d1b043aff24fb54f3aa3be010");
}
TEST(CanonicalHashTest, MatchesNormativeBytesVector) {
  auto hash = CanonicalHashBuilder::Create("pih:hash-golden:v1", 1);
  ASSERT_TRUE(hash.ok());
  const std::string_view abc = "abc";
  ASSERT_TRUE(hash->add_bytes(7, std::as_bytes(std::span(abc))).ok());
  EXPECT_EQ(hash->finalize().value().hex(),
            "807f4fcab3f684e7d7ced17be0a3bd36fe3dc4af65904037080f35b5b42aa577");
}
TEST(CanonicalHashTest, RejectsDomainOrderCountAndUnsafeUtf8) {
  EXPECT_FALSE(CanonicalHashBuilder::Create("other:v1", 0).ok());
  auto order = CanonicalHashBuilder::Create("pih:test:v1", 2).value();
  ASSERT_TRUE(order.add_u64(2, 1).ok());
  EXPECT_FALSE(order.add_u32(1, 1).ok());
  EXPECT_FALSE(order.finalize().ok());
  auto count = CanonicalHashBuilder::Create("pih:test:v1", 1).value();
  EXPECT_FALSE(count.finalize().ok());
  auto utf8 = CanonicalHashBuilder::Create("pih:test:v1", 1).value();
  EXPECT_FALSE(utf8.add_ascii_utf8(1, "\xc3\xa9").ok());
  auto zero = CanonicalHashBuilder::Create("pih:test:v1", 1).value();
  EXPECT_FALSE(zero.add_hash(1, Sha256Digest{}).ok());
}
}}  // namespace pih::<anonymous>
