#include "pih/model/engine_supervisor_shutdown_channel.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

class Driver final : public EngineSupervisorShutdownChannelDriver {
 public:
  Status send_request(std::span<const std::byte> frame) override {
    ++send_calls;
    sent.assign(frame.begin(), frame.end());
    return send_status;
  }

  Result<std::optional<std::vector<std::byte>>> poll_ack() override {
    ++poll_calls;
    if (!ack.has_value()) return std::optional<std::vector<std::byte>>{};
    return std::optional<std::vector<std::byte>>{*ack};
  }

  Status send_status = Status::Ok();
  std::optional<std::vector<std::byte>> ack;
  std::vector<std::byte> sent;
  int send_calls = 0;
  int poll_calls = 0;
};

EngineSupervisorShutdownRequest request() {
  return {7, 11, 1, 900, EngineSupervisorShutdownRequestKind::kForceStop};
}

std::vector<std::byte> ack(std::uint64_t epoch = 11) {
  const auto frame = encode_engine_supervisor_shutdown_ack(
      {7, epoch, 1, 800,
       EngineSupervisorShutdownAckDisposition::kAccepted});
  return {frame.begin(), frame.end()};
}

TEST(EngineSupervisorShutdownChannelTest, RetriesBackpressureThenAcceptsAck) {
  Driver driver;
  driver.send_status = Status::Unavailable("backpressured");
  auto channel = EngineSupervisorShutdownChannel::Create(request(), driver).value();
  EXPECT_FALSE(*channel.advance(800));
  EXPECT_EQ(driver.send_calls, 1);
  EXPECT_EQ(driver.poll_calls, 0);
  driver.send_status = Status::Ok();
  driver.ack = ack();
  EXPECT_TRUE(*channel.advance(801));
  EXPECT_TRUE(channel.request_acknowledged());
  EXPECT_EQ(driver.send_calls, 2);
  EXPECT_EQ(driver.poll_calls, 1);
  EXPECT_TRUE(*channel.advance(802));
  EXPECT_EQ(driver.send_calls, 2);
  EXPECT_EQ(driver.poll_calls, 1);
}

TEST(EngineSupervisorShutdownChannelTest, DeadlineEqualityPoisonsTransaction) {
  Driver driver;
  auto channel = EngineSupervisorShutdownChannel::Create(request(), driver).value();
  auto result = channel.advance(900);
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), StatusCode::kDeadlineExceeded);
  EXPECT_EQ(driver.send_calls, 0);
  EXPECT_FALSE(channel.advance(899).ok());
}

TEST(EngineSupervisorShutdownChannelTest, MalformedOrDriftedAckPoisonsTransaction) {
  Driver short_driver;
  short_driver.ack = std::vector<std::byte>(47);
  auto short_channel =
      EngineSupervisorShutdownChannel::Create(request(), short_driver).value();
  EXPECT_FALSE(short_channel.advance(800).ok());
  short_driver.ack = ack();
  EXPECT_FALSE(short_channel.advance(801).ok());

  Driver drift_driver;
  drift_driver.ack = ack(12);
  auto drift_channel =
      EngineSupervisorShutdownChannel::Create(request(), drift_driver).value();
  EXPECT_FALSE(drift_channel.advance(800).ok());
  EXPECT_FALSE(drift_channel.advance(801).ok());
}

}  // namespace
}  // namespace pih
