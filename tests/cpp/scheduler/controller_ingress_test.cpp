#include "pih/scheduler/controller_ingress.h"

#include <array>

#include <gtest/gtest.h>

namespace pih {
namespace {
ControllerIngress ingress(std::uint32_t command_capacity = 2) {
  return ControllerIngress::Create(
      {{command_capacity, 2}, {2, 4, 8}}).value();
}
}

TEST(ControllerIngressTest, AtomicallyPublishesCommandAndImmutablePayload) {
  auto port = ingress();
  const std::array<std::uint32_t, 3> tokens{1, 2, 3};
  auto submitted = port.submit_admit(7, 11, tokens, 2);
  ASSERT_TRUE(submitted.ok()) << submitted.status().message();
  EXPECT_EQ(port.queued_command_count(), 1U);
  EXPECT_EQ(port.published_request_count(), 1U);
  auto admission = port.try_take_admission().value();
  ASSERT_TRUE(admission.has_value());
  EXPECT_EQ(admission->command.command_sequence, 1U);
  EXPECT_EQ(admission->request.request_generation, 11U);
  EXPECT_EQ(admission->request.prompt_token_ids[2], 3U);
  EXPECT_EQ(admission->request.maximum_new_tokens, 2U);
  EXPECT_EQ(admission->command.payload_digest,
            admission->request.payload_digest);
  EXPECT_EQ(port.claimed_request_count(), 1U);
  EXPECT_TRUE(port.release_request(admission->request).ok());
  EXPECT_EQ(port.claimed_request_count(), 0U);
}

TEST(ControllerIngressTest, MailboxFailureRollsBackAndScrubsRequestCredit) {
  auto port = ingress(1);
  const std::array<std::uint32_t, 1> first{4};
  const std::array<std::uint32_t, 2> second{8, 9};
  ASSERT_TRUE(port.submit_admit(1, 1, first, 1).ok());
  EXPECT_FALSE(port.submit_admit(1, 2, second, 1).ok());
  EXPECT_EQ(port.queued_command_count(), 1U);
  EXPECT_EQ(port.published_request_count(), 1U);
  auto admission = port.try_take_admission().value();
  ASSERT_TRUE(admission.has_value());
  EXPECT_EQ(admission->request.request_generation, 1U);
  EXPECT_TRUE(port.release_request(admission->request).ok());
  EXPECT_TRUE(port.submit_admit(1, 2, second, 1).ok());
}

TEST(ControllerIngressTest, EmptyPollDoesNotChangeControllerOwnershipOrCredits) {
  auto port = ingress();
  EXPECT_FALSE(port.try_take_admission().value().has_value());
  EXPECT_EQ(port.queued_command_count(), 0U);
  EXPECT_EQ(port.published_request_count(), 0U);
}

TEST(ControllerIngressTest, EventPathRetainsMailboxBackpressure) {
  auto port = ingress();
  ASSERT_FALSE(port.try_take_admission().value().has_value());
  constexpr std::string_view value = "state";
  const auto state = sha256(std::as_bytes(std::span(value))).value();
  auto event = port.publish_event(
      {ControllerOutputEventKind::kAdmitted, 1, 1, 0, 0, state});
  ASSERT_TRUE(event.ok());
  auto observed = port.try_take_event().value();
  ASSERT_TRUE(observed.has_value());
  EXPECT_EQ(observed->event_sequence, event->event_sequence);
}

TEST(ControllerIngressTest, CancelCommandHasCanonicalPayloadWithoutRequestSlot) {
  auto port = ingress();
  auto submitted = port.submit_cancel(7, 11);
  ASSERT_TRUE(submitted.ok()) << submitted.status().message();
  auto expected = controller_cancel_payload_digest(7, 11).value();
  EXPECT_EQ(submitted->payload_digest, expected);
  EXPECT_EQ(port.published_request_count(), 0U);
  auto item = port.try_take_command().value();
  ASSERT_TRUE(item.has_value());
  EXPECT_EQ(item->command.kind, ControllerCommandKind::kCancel);
  EXPECT_FALSE(item->request.has_value());
}

}  // namespace pih
