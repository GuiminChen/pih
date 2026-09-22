#include "pih/io/durable_file_sink.h"

#include <array>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

namespace pih {
namespace {

class DurableFileSinkTest : public ::testing::Test {
 protected:
  void SetUp() override {
    path_ = std::filesystem::temp_directory_path() /
            ("pih-durable-sink-" + std::to_string(counter_++));
    std::filesystem::remove(path_);
  }
  void TearDown() override { std::filesystem::remove(path_); }
  std::filesystem::path path_;
  static inline std::uint64_t counter_ = 1;
};

TEST_F(DurableFileSinkTest, ExclusivelyWritesExactDurableLength) {
  auto sink = DurableFileSink::CreateExclusive(path_, 5);
  ASSERT_TRUE(sink.ok()) << sink.status().message();
  const std::array first{std::byte{'a'}, std::byte{'b'}};
  const std::array second{std::byte{'c'}, std::byte{'d'}, std::byte{'e'}};
  EXPECT_EQ(sink->write(first).value(), 2U);
  EXPECT_EQ(sink->write(second).value(), 3U);
  EXPECT_TRUE(sink->sync().ok());
  EXPECT_FALSE(sink->write(first).ok());
  std::ifstream input(path_, std::ios::binary);
  EXPECT_EQ(std::string(std::istreambuf_iterator<char>(input), {}), "abcde");
}

TEST_F(DurableFileSinkTest, RejectsOverwriteOverflowAndIncompleteSync) {
  auto sink = DurableFileSink::CreateExclusive(path_, 2);
  ASSERT_TRUE(sink.ok());
  EXPECT_FALSE(DurableFileSink::CreateExclusive(path_, 2).ok());
  const std::array oversized{std::byte{1}, std::byte{2}, std::byte{3}};
  EXPECT_FALSE(sink->write(oversized).ok());
  EXPECT_FALSE(sink->sync().ok());
}

}  // namespace
}  // namespace pih
