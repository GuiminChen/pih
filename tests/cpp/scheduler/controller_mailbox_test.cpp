#include "pih/scheduler/controller_mailbox.h"

#include <future>

#include <gtest/gtest.h>

namespace pih {
namespace {
Sha256Digest digest(std::string_view value) {
  return sha256(std::as_bytes(std::span(value))).value();
}
}

TEST(ControllerMailboxTest, PreservesBoundedFifoCommandIdentity) {
  auto mailbox = ControllerMailbox::Create({2, 2}).value();
  auto first = mailbox.submit_command(
      ControllerCommandKind::kAdmit, 1, 11, digest("a"));
  auto second = mailbox.submit_command(
      ControllerCommandKind::kCancel, 1, 11, digest("b"));
  ASSERT_TRUE(first.ok());
  ASSERT_TRUE(second.ok());
  EXPECT_EQ(first->command_sequence, 1U);
  EXPECT_EQ(second->command_sequence, 2U);
  EXPECT_FALSE(mailbox.submit_command(
      ControllerCommandKind::kShutdown, 1, 11, digest("c")).ok());
  auto taken_first = mailbox.try_take_command().value();
  auto taken_second = mailbox.try_take_command().value();
  ASSERT_TRUE(taken_first.has_value());
  ASSERT_TRUE(taken_second.has_value());
  EXPECT_EQ(taken_first->payload_digest, digest("a"));
  EXPECT_EQ(taken_second->payload_digest, digest("b"));
  EXPECT_FALSE(mailbox.try_take_command().value().has_value());
}

TEST(ControllerMailboxTest, PublishesImmutableBoundedEvents) {
  auto mailbox = ControllerMailbox::Create({1, 2}).value();
  ASSERT_FALSE(mailbox.try_take_command().value().has_value());
  auto admitted = mailbox.publish_event(
      {ControllerOutputEventKind::kAdmitted, 1, 9, 0, 0, digest("state-1")});
  auto token = mailbox.publish_event(
      {ControllerOutputEventKind::kTokenCommitted, 1, 9, 1, 42,
       digest("state-2"), 7, 0, 1});
  ASSERT_TRUE(admitted.ok());
  ASSERT_TRUE(token.ok());
  EXPECT_EQ(admitted->event_sequence, 1U);
  EXPECT_EQ(token->event_sequence, 2U);
  EXPECT_FALSE(mailbox.publish_event(
      {ControllerOutputEventKind::kCompleted, 1, 9, 0, 0,
       digest("state-3")}).ok());
  const auto first = mailbox.try_take_event().value();
  const auto second = mailbox.try_take_event().value();
  ASSERT_TRUE(first.has_value());
  ASSERT_TRUE(second.has_value());
  EXPECT_EQ(first->kind, ControllerOutputEventKind::kAdmitted);
  EXPECT_EQ(second->token_id, 42U);
}

TEST(ControllerMailboxTest, ControllerSideHasExactlyOneThreadOwner) {
  auto mailbox = ControllerMailbox::Create({1, 1}).value();
  ASSERT_FALSE(mailbox.try_take_command().value().has_value());
  auto result = std::async(std::launch::async, [&] {
    return mailbox.publish_event(
        {ControllerOutputEventKind::kAdmitted, 1, 1, 0, 0, digest("x")});
  }).get();
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(mailbox.event_size(), 0U);
}

TEST(ControllerMailboxTest, RejectsMalformedIdentityWithoutConsumingCredit) {
  auto mailbox = ControllerMailbox::Create({1, 1}).value();
  EXPECT_FALSE(mailbox.submit_command(
      ControllerCommandKind::kAdmit, 0, 1, digest("x")).ok());
  EXPECT_EQ(mailbox.command_size(), 0U);
  ASSERT_FALSE(mailbox.try_take_command().value().has_value());
  EXPECT_FALSE(mailbox.publish_event(
      {ControllerOutputEventKind::kTokenCommitted, 1, 1, 0, 151936,
       digest("x"), 7, 0, 1}).ok());
  EXPECT_EQ(mailbox.event_size(), 0U);
}

}  // namespace pih
