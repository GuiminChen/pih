#include <gtest/gtest.h>

// Exercise the actual private callback implementation, not a duplicate model.
// This executable owns the entrypoint; it does not load a plugin or an engine.
#include "../../../plugins/surface-text-http/entrypoint.cpp"

TEST(NativeTextOutputContract, RejectsInvalidStartModePermanently) {
  Output output{-1};
  OutputScope scope(output);
  EXPECT_EQ(StartOutput(&output, 2), 0U);
  EXPECT_TRUE(output.contract_failed);
  EXPECT_EQ(Cancelled(&output), 1U);
  EXPECT_FALSE(output.started);
}

TEST(NativeTextOutputContract, RejectsDuplicateStartPermanently) {
  Output output{-1};
  OutputScope scope(output);
  output.started = true;
  EXPECT_EQ(StartOutput(&output, 0), 0U);
  EXPECT_TRUE(output.contract_failed);
  EXPECT_EQ(Cancelled(&output), 1U);
}

TEST(NativeTextOutputContract, RejectsWriteBeforeStart) {
  Output output{-1};
  OutputScope scope(output);
  EXPECT_EQ(WriteOutput(&output, "{}", 2), 0U);
  EXPECT_TRUE(output.contract_failed);
  EXPECT_EQ(Cancelled(&output), 1U);
}

TEST(NativeTextOutputContract, RejectsStreamingWriteInBufferedMode) {
  Output output{-1};
  OutputScope scope(output);
  output.started = true;
  EXPECT_EQ(WriteOutput(&output, "{}", 2), 0U);
  EXPECT_TRUE(output.contract_failed);
  EXPECT_EQ(Cancelled(&output), 1U);
}

TEST(NativeTextOutputContract, RejectsInjectedSseLinePermanently) {
  int descriptors[2];
  ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC,
                       0, descriptors), 0);
  Descriptor sender(descriptors[0]);
  Descriptor receiver(descriptors[1]);
  Output output{sender.value};
  OutputScope scope(output);
  // A live peer with no pending input lets cancellation inspection succeed.
  output.started = true;
  output.streaming = true;
  EXPECT_EQ(WriteOutput(&output, "{}\n", 3), 0U);
  EXPECT_TRUE(output.contract_failed);
  EXPECT_EQ(Cancelled(&output), 1U);
  EXPECT_EQ(WriteOutput(&output, "{}", 2), 0U);
  char byte;
  EXPECT_EQ(::recv(receiver.value, &byte, 1, MSG_DONTWAIT), -1);
  EXPECT_TRUE(errno == EAGAIN || errno == EWOULDBLOCK);
}

TEST(NativeTextOutputContract, RejectsForeignCallbackContext) {
  Output output{-1};
  Output foreign{-1};
  OutputScope scope(output);
  EXPECT_EQ(StartOutput(&foreign, 0), 0U);
  EXPECT_TRUE(output.contract_failed);
  EXPECT_FALSE(foreign.started);
}

TEST(NativeTextOutputContract, RejectsNullCallbackContext) {
  Output output{-1};
  OutputScope scope(output);
  EXPECT_EQ(StartOutput(nullptr, 0), 0U);
  EXPECT_TRUE(output.contract_failed);
  EXPECT_EQ(Cancelled(&output), 1U);
}

TEST(NativeTextOutputContract, RejectsMalformedJsonEvent) {
  int descriptors[2];
  ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC,
                       0, descriptors), 0);
  Descriptor sender(descriptors[0]);
  Descriptor receiver(descriptors[1]);
  Output output{sender.value};
  OutputScope scope(output);
  output.started = true;
  output.streaming = true;
  EXPECT_EQ(WriteOutput(&output, "{invalid}", 9), 0U);
  EXPECT_TRUE(output.contract_failed);
  char byte;
  EXPECT_EQ(::recv(receiver.value, &byte, 1, MSG_DONTWAIT), -1);
  EXPECT_TRUE(errno == EAGAIN || errno == EWOULDBLOCK);
}
