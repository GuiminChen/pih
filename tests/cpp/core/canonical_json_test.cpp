#include "pih/core/canonical_json.h"

#include <array>

#include <gtest/gtest.h>

namespace pih {
namespace {

TEST(CanonicalJsonTest, SortsAndEncodesExactAsciiAuthorityForm) {
  auto parsed = JsonValue::Parse(
      R"({"z":[true,null,-7],"body_sha256":"omit","a":"line\nquote\""})");
  ASSERT_TRUE(parsed.ok());
  const std::array<std::string_view, 1> omitted{"body_sha256"};
  auto encoded = canonical_ascii_json(*parsed, 1024, omitted);
  ASSERT_TRUE(encoded.ok()) << encoded.status().message();
  EXPECT_EQ(*encoded, R"({"a":"line\nquote\"","z":[true,null,-7]})");
}

TEST(CanonicalJsonTest, FailsClosedOnUnsupportedOrUnboundedAuthority) {
  auto floating = JsonValue::Parse(R"({"value":1.5})").value();
  EXPECT_FALSE(canonical_ascii_json(floating, 1024).ok());
  auto non_ascii = JsonValue::Parse(R"({"value":"\u4e2d"})").value();
  EXPECT_FALSE(canonical_ascii_json(non_ascii, 1024).ok());
  auto value = JsonValue::Parse(R"({"value":"abcdef"})").value();
  EXPECT_FALSE(canonical_ascii_json(value, 4).ok());
}

}  // namespace
}  // namespace pih
