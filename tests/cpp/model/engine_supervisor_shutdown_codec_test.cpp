#include "pih/model/engine_supervisor_shutdown_codec.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

EngineSupervisorShutdownRequest request() {
  return {7, 11, 1, 900, EngineSupervisorShutdownRequestKind::kForceStop};
}

EngineSupervisorShutdownAck ack() {
  return {7, 11, 1, 850,
          EngineSupervisorShutdownAckDisposition::kAccepted};
}

TEST(EngineSupervisorShutdownCodecTest, RoundTripsCanonicalRequestAndAck) {
  const auto request_bytes = encode_engine_supervisor_shutdown_request(request());
  ASSERT_EQ(request_bytes.size(), kEngineSupervisorShutdownFrameBytes);
  const auto decoded_request =
      decode_engine_supervisor_shutdown_request(request_bytes);
  ASSERT_TRUE(decoded_request.ok());
  EXPECT_EQ(decoded_request->engine_generation, 7U);
  EXPECT_EQ(decoded_request->engine_epoch, 11U);
  EXPECT_EQ(decoded_request->request_identity, 1U);
  EXPECT_EQ(decoded_request->deadline_ns, 900U);
  EXPECT_EQ(decoded_request->kind,
            EngineSupervisorShutdownRequestKind::kForceStop);

  const auto ack_bytes = encode_engine_supervisor_shutdown_ack(ack());
  const auto decoded_ack = decode_engine_supervisor_shutdown_ack(ack_bytes);
  ASSERT_TRUE(decoded_ack.ok());
  EXPECT_EQ(decoded_ack->acknowledged_ns, 850U);
  EXPECT_EQ(decoded_ack->disposition,
            EngineSupervisorShutdownAckDisposition::kAccepted);
}

TEST(EngineSupervisorShutdownCodecTest, RejectsSizeHeaderReservedAndDomainDrift) {
  auto bytes = encode_engine_supervisor_shutdown_request(request());
  EXPECT_FALSE(decode_engine_supervisor_shutdown_request(
      std::span<const std::byte>(bytes).first(bytes.size() - 1)).ok());
  bytes[0] = std::byte{0};
  EXPECT_FALSE(decode_engine_supervisor_shutdown_request(bytes).ok());
  bytes = encode_engine_supervisor_shutdown_request(request());
  bytes[41] = std::byte{1};
  EXPECT_FALSE(decode_engine_supervisor_shutdown_request(bytes).ok());

  auto invalid = request();
  invalid.engine_epoch = 0;
  EXPECT_FALSE(decode_engine_supervisor_shutdown_request(
      encode_engine_supervisor_shutdown_request(invalid)).ok());
}

TEST(EngineSupervisorShutdownCodecTest, AckGateAcceptsExactlyOneBoundSuccess) {
  auto gate = EngineSupervisorShutdownAckGate::Create(7, 11, 1).value();
  EXPECT_TRUE(gate.accept(ack()).ok());
  EXPECT_TRUE(gate.accepted());
  EXPECT_FALSE(gate.accept(ack()).ok());

  auto mismatch = EngineSupervisorShutdownAckGate::Create(7, 11, 1).value();
  auto wrong = ack();
  wrong.engine_epoch = 12;
  EXPECT_FALSE(mismatch.accept(wrong).ok());
  EXPECT_FALSE(mismatch.accept(ack()).ok());

  auto rejected = EngineSupervisorShutdownAckGate::Create(7, 11, 1).value();
  auto refusal = ack();
  refusal.disposition = EngineSupervisorShutdownAckDisposition::kRejected;
  EXPECT_FALSE(rejected.accept(refusal).ok());
  EXPECT_FALSE(rejected.accepted());
}

TEST(EngineSupervisorShutdownCodecTest, RequestGateDeduplicatesExactReplay) {
  auto gate = EngineSupervisorShutdownRequestGate::Create(7, 11).value();
  EXPECT_EQ(*gate.accept(request(), 800),
            EngineSupervisorShutdownAckDisposition::kAccepted);
  EXPECT_EQ(*gate.accept(request(), 801),
            EngineSupervisorShutdownAckDisposition::kAlreadyStopping);
  EXPECT_TRUE(gate.accepted());
}

TEST(EngineSupervisorShutdownCodecTest, RequestGatePoisonsDriftAndExpiry) {
  auto drift = EngineSupervisorShutdownRequestGate::Create(7, 11).value();
  auto wrong = request();
  wrong.engine_generation = 8;
  EXPECT_FALSE(drift.accept(wrong, 800).ok());
  EXPECT_FALSE(drift.accept(request(), 801).ok());

  auto mutation = EngineSupervisorShutdownRequestGate::Create(7, 11).value();
  ASSERT_TRUE(mutation.accept(request(), 800).ok());
  wrong = request();
  wrong.deadline_ns = 901;
  EXPECT_FALSE(mutation.accept(wrong, 801).ok());
  EXPECT_FALSE(mutation.accept(request(), 802).ok());

  auto expired = EngineSupervisorShutdownRequestGate::Create(7, 11).value();
  EXPECT_FALSE(expired.accept(request(), 900).ok());
  EXPECT_FALSE(expired.accept(request(), 899).ok());
}

}  // namespace
}  // namespace pih
