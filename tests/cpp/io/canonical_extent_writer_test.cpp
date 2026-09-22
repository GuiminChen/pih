#include "pih/io/canonical_extent_writer.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace pih {
namespace {

class Reader final : public CanonicalPayloadReader {
 public:
  std::map<std::string, std::string> payloads;
  std::size_t maximum_read = 3;
  Result<std::size_t> read(std::string_view identity, std::uint64_t offset,
                           std::span<std::byte> output) override {
    const auto found = payloads.find(std::string(identity));
    if (found == payloads.end() || offset >= found->second.size()) return 0U;
    const auto count = std::min({maximum_read, output.size(),
                                 found->second.size() - static_cast<std::size_t>(offset)});
    std::memcpy(output.data(), found->second.data() + offset, count);
    return count;
  }
};

class Sink final : public CanonicalSequentialSink {
 public:
  std::vector<std::byte> bytes;
  std::size_t maximum_write = 2;
  bool fail_sync = false;
  Result<std::size_t> write(std::span<const std::byte> input) override {
    const auto count = std::min(maximum_write, input.size());
    bytes.insert(bytes.end(), input.begin(), input.begin() + count);
    return count;
  }
  Status sync() override {
    return fail_sync ? Status::Internal("injected sync failure") : Status::Ok();
  }
};

TEST(CanonicalExtentWriterTest, StreamsPartialIoAndZeroFillsEveryPaddingByte) {
  const std::string metadata = "meta";
  const std::vector<CanonicalExtentWritePlan> extents{
      {"a", 8, 3, 4}, {"b", 12, 5, 8}};
  Reader reader;
  reader.payloads = {{"a", "abc"}, {"b", "12345"}};
  Sink sink;
  auto receipt = write_canonical_extents(
      std::as_bytes(std::span(metadata)), 8, extents, reader, sink, 3);
  ASSERT_TRUE(receipt.ok()) << receipt.status().message();
  ASSERT_EQ(sink.bytes.size(), 20U);
  EXPECT_EQ(std::string(reinterpret_cast<const char*>(sink.bytes.data()), 4), "meta");
  for (const auto index : {4U, 5U, 6U, 7U, 11U, 17U, 18U, 19U}) {
    EXPECT_EQ(sink.bytes[index], std::byte{0});
  }
  EXPECT_EQ(receipt->file_bytes, 20U);
  EXPECT_EQ(receipt->maximum_workspace_bytes, 3U);
  EXPECT_EQ(receipt->file_sha256, sha256(sink.bytes).value());
}

TEST(CanonicalExtentWriterTest, RejectsLayoutAndZeroProgress) {
  Reader reader;
  reader.payloads = {{"a", "abc"}};
  Sink sink;
  const std::vector<CanonicalExtentWritePlan> gap{{"a", 9, 3, 4}};
  EXPECT_FALSE(write_canonical_extents({}, 8, gap, reader, sink, 3).ok());
  const std::vector<CanonicalExtentWritePlan> valid{{"a", 8, 4, 4}};
  EXPECT_FALSE(write_canonical_extents({}, 8, valid, reader, sink, 3).ok());
}

TEST(CanonicalExtentWriterTest, DoesNotSealBeforeDurableSync) {
  Reader reader;
  reader.payloads = {{"a", "abc"}};
  Sink sink;
  sink.fail_sync = true;
  const std::vector<CanonicalExtentWritePlan> extents{{"a", 4, 3, 4}};
  EXPECT_FALSE(write_canonical_extents({}, 4, extents, reader, sink, 2).ok());
}

}  // namespace
}  // namespace pih
