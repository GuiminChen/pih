#include "../../../plugins/model-qwen3/text_stop.h"
#include <gtest/gtest.h>

namespace pih::qwen_plugin {
TEST(QwenNativeTextStop, RetainsSplitMarkerAndDropsTail) {
  JsonValue stop(std::string("END"));
  TextStop filter(&stop);
  EXPECT_EQ(filter.Feed("hello E"), "hello ");
  EXPECT_EQ(filter.Feed("N"), "");
  EXPECT_EQ(filter.Feed("D hidden"), "");
  EXPECT_TRUE(filter.matched());
  EXPECT_EQ(filter.Finish(), "");
}
TEST(QwenNativeTextStop, FlushesUnmatchedPrefix) {
  JsonValue stop(std::string("END"));
  TextStop filter(&stop);
  EXPECT_EQ(filter.Feed("hello EN"), "hello ");
  EXPECT_EQ(filter.Finish(), "EN");
  EXPECT_FALSE(filter.matched());
  EXPECT_THROW(filter.Feed("more"), std::invalid_argument);
}
TEST(QwenNativeTextStop, OverlapCompletionIsChunkIndependent) {
  JsonValue stop(JsonValue::Array{JsonValue(std::string("abcd")), JsonValue(std::string("bc"))});
  TextStop whole(&stop), split(&stop);
  EXPECT_EQ(whole.Feed("abcd"), "a");
  std::string observed;
  for (const auto part : {"a", "b", "c", "d"}) observed += split.Feed(part);
  EXPECT_EQ(observed, "a");
  EXPECT_TRUE(whole.matched());
  EXPECT_TRUE(split.matched());
}
TEST(QwenNativeTextStop, RejectsEmptyAndDuplicateMarkers) {
  JsonValue empty(std::string{});
  EXPECT_THROW((TextStop(&empty)), std::invalid_argument);
  JsonValue duplicate(JsonValue::Array{JsonValue(std::string("x")), JsonValue(std::string("x"))});
  EXPECT_THROW((TextStop(&duplicate)), std::invalid_argument);
}
TEST(QwenNativeTextStop, NullAndMissingStopPassThrough) {
  JsonValue null(nullptr);
  TextStop missing(nullptr), disabled(&null);
  EXPECT_EQ(missing.Feed("plain text"), "plain text");
  EXPECT_EQ(disabled.Feed("plain text"), "plain text");
  EXPECT_EQ(missing.Finish(), "");
  EXPECT_EQ(disabled.Finish(), "");
}
TEST(QwenNativeTextStop, HandlesUnicodeMarkerAcrossCompleteFragments) {
  JsonValue stop(std::string("\xe7\xbb\x93\xe6\x9d\x9f"));
  TextStop filter(&stop);
  EXPECT_EQ(filter.Feed("answer\xe7\xbb\x93"), "answer");
  EXPECT_EQ(filter.Feed("\xe6\x9d\x9f trailing"), "");
  EXPECT_TRUE(filter.matched());
  EXPECT_EQ(filter.Feed("ignored"), "");
  EXPECT_EQ(filter.Finish(), "");
}
TEST(QwenNativeTextStop, EarliestEndTieUsesEarliestStart) {
  JsonValue stop(JsonValue::Array{JsonValue(std::string("bc")), JsonValue(std::string("abc"))});
  TextStop filter(&stop);
  EXPECT_EQ(filter.Feed("xabc tail"), "x");
  EXPECT_TRUE(filter.matched());
}
TEST(QwenNativeTextStop, RejectsTooManyAndOversizedMarkers) {
  JsonValue oversized(std::string(257, 'x'));
  EXPECT_THROW((TextStop(&oversized)), std::invalid_argument);
  JsonValue many(JsonValue::Array{JsonValue(std::string("a")), JsonValue(std::string("b")),
      JsonValue(std::string("c")), JsonValue(std::string("d")), JsonValue(std::string("e"))});
  EXPECT_THROW((TextStop(&many)), std::invalid_argument);
}
}  // namespace pih::qwen_plugin
