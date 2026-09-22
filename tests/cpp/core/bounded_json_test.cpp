#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "pih/core/bounded_json.h"

namespace pih {
namespace {

TEST(BoundedJsonTest, ParsesTypedObjectWithoutCoercion) {
  auto parsed = JsonValue::Parse(
      R"({"name":"qwen\u0033","layers":28,"epsilon":1e-6,"tied":true,"shape":[151936,1024],"none":null})");
  ASSERT_TRUE(parsed.ok()) << parsed.status().message();
  ASSERT_TRUE(parsed->is_object());
  EXPECT_EQ(parsed->at("name")->string(), "qwen3");
  EXPECT_EQ(parsed->at("layers")->integer(), 28);
  EXPECT_DOUBLE_EQ(parsed->at("epsilon")->number(), 1e-6);
  EXPECT_TRUE(parsed->at("tied")->boolean());
  ASSERT_EQ(parsed->at("shape")->array().size(), 2);
  EXPECT_TRUE(parsed->at("none")->is_null());
}

TEST(BoundedJsonTest, RejectsDuplicatesTrailingDataAndInvalidNumbers) {
  EXPECT_FALSE(JsonValue::Parse(R"({"a":1,"a":2})").ok());
  EXPECT_FALSE(JsonValue::Parse(R"({"a":1}x)").ok());
  EXPECT_FALSE(JsonValue::Parse(R"({"a":01})").ok());
  EXPECT_FALSE(JsonValue::Parse(R"({"a":NaN})").ok());
  EXPECT_FALSE(JsonValue::Parse(R"({"a":9223372036854775808})").ok());
}

TEST(BoundedJsonTest, EnforcesInputDepthNodeAndStringBudgets) {
  JsonLimits limits;
  limits.max_input_bytes = 8;
  EXPECT_FALSE(JsonValue::Parse(R"({"long":1})", limits).ok());

  limits = JsonLimits{};
  limits.max_depth = 2;
  EXPECT_FALSE(JsonValue::Parse(R"({"a":{"b":1}})", limits).ok());

  limits = JsonLimits{};
  limits.max_nodes = 3;
  EXPECT_FALSE(JsonValue::Parse(R"([1,2,3,4])", limits).ok());

  limits = JsonLimits{};
  limits.max_string_bytes = 3;
  EXPECT_FALSE(JsonValue::Parse(R"("four")", limits).ok());
}

TEST(BoundedJsonTest, RejectsMalformedUnicodeAndControlCharacters) {
  EXPECT_FALSE(JsonValue::Parse(R"("\uD800")").ok());
  EXPECT_FALSE(JsonValue::Parse(std::string("\"a\n\"", 4)).ok());
}

TEST(BoundedJsonTest, AcceptsOnlyCanonicalRawUtf8ScalarSequences) {
  const std::vector<std::string> valid{
      std::string("\xC2\x80", 2),
      std::string("\xDF\xBF", 2),
      std::string("\xE0\xA0\x80", 3),
      std::string("\xED\x9F\xBF", 3),
      std::string("\xEE\x80\x80", 3),
      std::string("\xEF\xBF\xBF", 3),
      std::string("\xF0\x90\x80\x80", 4),
      std::string("\xF4\x8F\xBF\xBF", 4),
  };
  for (std::size_t index = 0; index < valid.size(); ++index) {
    SCOPED_TRACE(index);
    const auto input = std::string("\"") + valid[index] + "\"";
    auto parsed = JsonValue::Parse(input);
    ASSERT_TRUE(parsed.ok()) << parsed.status().message();
    EXPECT_EQ(parsed->string(), valid[index]);
  }

  const std::string raw_e_acute("\xC3\xA9", 2);
  const auto duplicate =
      std::string("{\"") + raw_e_acute + R"(":1,"\u00E9":2})";
  EXPECT_FALSE(JsonValue::Parse(duplicate).ok());
}

TEST(BoundedJsonTest, RejectsMalformedRawUtf8Sequences) {
  const std::vector<std::string> invalid{
      std::string("\x80", 1),
      std::string("\xBF", 1),
      std::string("\xC0\x80", 2),
      std::string("\xC1\xBF", 2),
      std::string("\xC2", 1),
      std::string("\xC2\x20", 2),
      std::string("\xE0\x80\x80", 3),
      std::string("\xED\xA0\x80", 3),
      std::string("\xE1\x80", 2),
      std::string("\xE1\x28\xA1", 3),
      std::string("\xF0\x80\x80\x80", 4),
      std::string("\xF4\x90\x80\x80", 4),
      std::string("\xF5\x80\x80\x80", 4),
      std::string("\xF1\x80\x80", 3),
      std::string("\xFF", 1),
  };
  for (std::size_t index = 0; index < invalid.size(); ++index) {
    SCOPED_TRACE(index);
    const auto input = std::string("\"") + invalid[index] + "\"";
    const auto parsed = JsonValue::Parse(input);
    ASSERT_FALSE(parsed.ok());
    EXPECT_EQ(parsed.status().code(), StatusCode::kInvalidArgument);
  }
}

TEST(BoundedJsonTest, AppliesDecodedStringBudgetToRawAndEscapedUtf8) {
  JsonLimits limits;
  limits.max_string_bytes = 3;
  const auto raw_over_budget =
      JsonValue::Parse(std::string("\"\xF0\x90\x80\x80\"", 6), limits);
  const auto escaped_over_budget =
      JsonValue::Parse(R"("\uD800\uDC00")", limits);
  ASSERT_FALSE(raw_over_budget.ok());
  ASSERT_FALSE(escaped_over_budget.ok());
  EXPECT_EQ(raw_over_budget.status().code(), StatusCode::kResourceExhausted);
  EXPECT_EQ(escaped_over_budget.status().code(),
            StatusCode::kResourceExhausted);

  limits.max_string_bytes = 4;
  auto raw =
      JsonValue::Parse(std::string("\"\xF0\x90\x80\x80\"", 6), limits);
  auto escaped = JsonValue::Parse(R"("\uD800\uDC00")", limits);
  ASSERT_TRUE(raw.ok()) << raw.status().message();
  ASSERT_TRUE(escaped.ok()) << escaped.status().message();
  EXPECT_EQ(raw->string(), escaped->string());

  const std::string raw_key("\xF0\x90\x80\x80", 4);
  const auto duplicate =
      std::string("{\"") + raw_key + R"(":1,"\uD800\uDC00":2})";
  EXPECT_FALSE(JsonValue::Parse(duplicate).ok());
}

}  // namespace
}  // namespace pih
