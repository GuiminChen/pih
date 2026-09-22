#include "pih/scheduler/output_slot_arena.h"

#include <array>

#include <gtest/gtest.h>

namespace pih {
namespace {

OutputSlotArena arena() {
  return OutputSlotArena::Create({2, 3, 2, 8, 4, 2}).value();
}

TEST(OutputSlotArenaTest, ReservesChainAndScrubsOnlyAfterCompleteWrite) {
  auto output = arena();
  const auto sequence = output.admit(11).value();
  const std::array<std::byte, 9> frame{};
  auto lease = output.reserve_frame(sequence, 11, frame, false).value();
  EXPECT_EQ(lease.slot_count, 2U);
  EXPECT_EQ(output.free_shared_slots(), 1U);
  ASSERT_TRUE(output.publish(lease).ok());
  ASSERT_TRUE(output.advance_writer(lease, 4).ok());
  EXPECT_EQ(output.state_count(OutputSlotState::kWriterOwned), 2U);
  ASSERT_TRUE(output.advance_writer(lease, 5).ok());
  EXPECT_EQ(output.state_count(OutputSlotState::kScrubPending), 2U);
  EXPECT_FALSE(output.release_sequence(sequence, 11).ok());
  ASSERT_TRUE(output.scrub(lease).ok());
  EXPECT_EQ(output.free_shared_slots(), 3U);
  EXPECT_TRUE(output.release_sequence(sequence, 11).ok());
}

TEST(OutputSlotArenaTest, TerminalReserveIsPerSequenceAndNeverBorrowed) {
  auto output = arena();
  const auto first = output.admit(11).value();
  const auto second = output.admit(12).value();
  const std::array<std::byte, 16> terminal{};
  auto first_lease = output.reserve_frame(first, 11, terminal, true).value();
  EXPECT_FALSE(output.reserve_frame(first, 11, std::array<std::byte, 1>{}, true).ok());
  auto second_lease = output.reserve_frame(second, 12, terminal, true).value();
  EXPECT_EQ(output.free_shared_slots(), 3U);
  EXPECT_NE(first_lease.slot_indices[0], second_lease.slot_indices[0]);
}

TEST(OutputSlotArenaTest, BuildReservationCanRollbackButPublishedLeaseCannot) {
  auto output = arena();
  const auto sequence = output.admit(11).value();
  const std::array<std::byte, 9> bytes{};
  const auto lease = output.reserve_frame(sequence, 11, bytes, false).value();
  EXPECT_EQ(output.free_shared_slots(), 1U);
  EXPECT_TRUE(output.abort_build(lease).ok());
  EXPECT_EQ(output.free_shared_slots(), 3U);
  EXPECT_FALSE(output.abort_build(lease).ok());

  const auto published = output.reserve_frame(sequence, 11, bytes, false).value();
  ASSERT_TRUE(output.publish(published).ok());
  EXPECT_FALSE(output.abort_build(published).ok());
}

TEST(OutputSlotArenaTest, DynamicCeilingAndGlobalOneShortFailBeforeOwnership) {
  auto output = arena();
  const auto first = output.admit(11).value();
  const auto second = output.admit(12).value();
  const std::array<std::byte, 16> two_slots{};
  ASSERT_TRUE(output.reserve_frame(first, 11, two_slots, false).ok());
  EXPECT_FALSE(output.reserve_frame(first, 11, std::array<std::byte, 1>{}, false).ok());
  EXPECT_FALSE(output.reserve_frame(second, 12, two_slots, false).ok());
  EXPECT_EQ(output.state_count(OutputSlotState::kBuildOwned), 2U);
}

TEST(OutputSlotArenaTest, OrphanRetainsChargeAndStaleLeaseCannotScrubReuse) {
  auto output = arena();
  const auto sequence = output.admit(11).value();
  const std::array<std::byte, 1> frame{};
  auto lease = output.reserve_frame(sequence, 11, frame, false).value();
  ASSERT_TRUE(output.publish(lease).ok());
  ASSERT_TRUE(output.orphan(lease).ok());
  EXPECT_EQ(output.free_shared_slots(), 2U);
  EXPECT_FALSE(output.scrub(lease).ok());
  ASSERT_TRUE(output.finish_orphan(lease).ok());
  ASSERT_TRUE(output.scrub(lease).ok());
  auto next = output.reserve_frame(sequence, 11, frame, false).value();
  EXPECT_NE(next.lease_generation, lease.lease_generation);
  EXPECT_FALSE(output.publish(lease).ok());
}

TEST(OutputSlotArenaTest, ForgedFrameLengthAndDuplicateChainFailClosed) {
  auto output = arena();
  const auto sequence = output.admit(11).value();
  const std::array<std::byte, 9> frame{};
  auto lease = output.reserve_frame(sequence, 11, frame, false).value();
  auto forged_length = lease;
  ++forged_length.frame_bytes;
  EXPECT_FALSE(output.publish(forged_length).ok());
  auto forged_chain = lease;
  forged_chain.slot_indices[1] = forged_chain.slot_indices[0];
  EXPECT_FALSE(output.publish(forged_chain).ok());
  EXPECT_TRUE(output.publish(lease).ok());
}

TEST(OutputSlotArenaTest, RejectsInvalidGeometry) {
  EXPECT_FALSE(OutputSlotArena::Create({0, 1, 1, 1, 1, 1}).ok());
  EXPECT_FALSE(OutputSlotArena::Create({1, 1, 1, 1, 65, 1}).ok());
}

}  // namespace
}  // namespace pih
