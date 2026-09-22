#include "pih/model/engine_inet_diag_multipart_decoder.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

EngineInetDiagMultipartMessage diag_row(
    std::uint32_t sequence, std::uint16_t port,
    EngineNetworkSocketLifecycle lifecycle) {
  return {sequence, EngineInetDiagMultipartMessageKind::kRow, true, false,
          0, {port, lifecycle, true, 64}};
}

EngineInetDiagMultipartMessage diag_done(std::uint32_t sequence) {
  return {sequence, EngineInetDiagMultipartMessageKind::kDone, true, false,
          0, {}};
}

TEST(EngineInetDiagMultipartDecoderTest, PublishesOnlyAfterMatchingDone) {
  auto decoder = EngineInetDiagMultipartDecoder::Create(7).value();
  EXPECT_TRUE(decoder.accept(diag_row(
      7, 40000, EngineNetworkSocketLifecycle::kActive)).ok());
  EXPECT_TRUE(decoder.accept(diag_row(
      7, 40001, EngineNetworkSocketLifecycle::kTimeWait)).ok());
  EXPECT_FALSE(decoder.finish().ok());
  EXPECT_TRUE(decoder.accept(diag_done(7)).ok());
  auto rows = decoder.finish();
  ASSERT_TRUE(rows.ok());
  ASSERT_EQ(rows->size(), 2U);
  EXPECT_EQ(rows->at(1).local_port, 40001);
}

TEST(EngineInetDiagMultipartDecoderTest, RejectsWrongSequenceAndTruncation) {
  for (int mutation = 0; mutation < 2; ++mutation) {
    auto decoder = EngineInetDiagMultipartDecoder::Create(7).value();
    auto message = diag_row(7, 40000,
                            EngineNetworkSocketLifecycle::kActive);
    if (mutation == 0) ++message.sequence;
    if (mutation == 1) message.truncated = true;
    auto status = decoder.accept(message);
    EXPECT_FALSE(status.ok()) << mutation;
    EXPECT_FALSE(decoder.finish().ok()) << mutation;
  }
}

TEST(EngineInetDiagMultipartDecoderTest, RejectsNetlinkError) {
  auto decoder = EngineInetDiagMultipartDecoder::Create(7).value();
  EngineInetDiagMultipartMessage error{
      7, EngineInetDiagMultipartMessageKind::kError, true, false, -13, {}};
  auto status = decoder.accept(error);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), StatusCode::kUnavailable);
  EXPECT_FALSE(decoder.finish().ok());
}

TEST(EngineInetDiagMultipartDecoderTest, RejectsMalformedRows) {
  for (int mutation = 0; mutation < 4; ++mutation) {
    auto decoder = EngineInetDiagMultipartDecoder::Create(7).value();
    auto message = diag_row(7, 40000,
                            EngineNetworkSocketLifecycle::kActive);
    if (mutation == 0) message.multipart = false;
    if (mutation == 1) message.row.local_port = 0;
    if (mutation == 2)
      message.row.lifecycle = static_cast<EngineNetworkSocketLifecycle>(2);
    if (mutation == 3) message.error_code = -1;
    EXPECT_FALSE(decoder.accept(message).ok()) << mutation;
  }
}

TEST(EngineInetDiagMultipartDecoderTest, RejectsMessagesAfterDone) {
  auto decoder = EngineInetDiagMultipartDecoder::Create(7).value();
  ASSERT_TRUE(decoder.accept(diag_done(7)).ok());
  EXPECT_FALSE(decoder.accept(diag_done(7)).ok());
  EXPECT_FALSE(decoder.accept(diag_row(
      7, 40000, EngineNetworkSocketLifecycle::kActive)).ok());
  EXPECT_FALSE(decoder.finish().ok());
  EXPECT_FALSE(EngineInetDiagMultipartDecoder::Create(0).ok());
}

}  // namespace
}  // namespace pih
