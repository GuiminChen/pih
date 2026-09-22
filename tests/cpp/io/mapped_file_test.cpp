#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>

#include <gtest/gtest.h>

#include "pih/io/mapped_file.h"

namespace pih {
namespace {

class MappedFileTest : public ::testing::Test {
 protected:
  void SetUp() override {
    path_ = std::filesystem::temp_directory_path() /
            ("pih-mapped-file-" + std::to_string(counter_++) + ".bin");
  }
  void TearDown() override { std::filesystem::remove(path_); }

  void write(std::string_view bytes) {
    std::ofstream stream(path_, std::ios::binary | std::ios::trunc);
    stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  }

  std::filesystem::path path_;
  static inline std::uint64_t counter_ = 1;
};

TEST_F(MappedFileTest, MapsReadOnlyAndReturnsBoundedSlices) {
  write("0123456789");
  auto mapped = MappedFile::OpenReadOnly(path_, 10);
  ASSERT_TRUE(mapped.ok()) << mapped.status().message();
  EXPECT_EQ(mapped->size_bytes(), 10);
  auto slice = mapped->slice(3, 4);
  ASSERT_TRUE(slice.ok());
  EXPECT_EQ(std::to_integer<char>((*slice)[0]), '3');
  EXPECT_EQ(std::to_integer<char>((*slice)[3]), '6');
  EXPECT_FALSE(mapped->slice(8, 3).ok());
}

TEST_F(MappedFileTest, MoveClearsSourceAndPreservesMapping) {
  write("abc");
  MappedFile first = std::move(MappedFile::OpenReadOnly(path_, 3)).value();
  MappedFile second = std::move(first);
  EXPECT_EQ(first.data(), nullptr);
  EXPECT_EQ(first.size_bytes(), 0);
  ASSERT_NE(second.data(), nullptr);
  EXPECT_EQ(std::to_integer<char>(second.data()[1]), 'b');
}

TEST_F(MappedFileTest, SupportsEmptyFileAndRejectsBudgetOrMissingPath) {
  write("");
  {
    auto empty = MappedFile::OpenReadOnly(path_, 0);
    ASSERT_TRUE(empty.ok());
    EXPECT_EQ(empty->data(), nullptr);
    EXPECT_EQ(empty->size_bytes(), 0);
  }

  write("too large");
  EXPECT_FALSE(MappedFile::OpenReadOnly(path_, 3).ok());
  std::filesystem::remove(path_);
  EXPECT_FALSE(MappedFile::OpenReadOnly(path_, 100).ok());
}

}  // namespace
}  // namespace pih
