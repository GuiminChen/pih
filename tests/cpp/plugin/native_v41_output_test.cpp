#include "../../../plugins/model-deepseek-v41/token_output.h"
#include <gtest/gtest.h>

namespace pih::deepseek_v41 {
namespace {
std::unique_ptr<TokenOutputQueue> Queue() {
  auto result = TokenOutputQueue::Create(1, sizeof(TokenPublication));
  if (!result.ok()) return nullptr;
  return std::move(*result);
}
TokenPublication Publication(std::uint64_t plan) {
  TokenPublication publication{};
  publication.record.observation.identity.plan_seq = plan;
  publication.record.accepted_count = 1;
  publication.visible[0] = 'x';
  publication.visible_size = 1;
  return publication;
}
TEST(NativeV41Output, PreparedReservationCannotBeReadOrReleased) {
  auto queue = Queue(); ASSERT_NE(queue, nullptr);
  auto lease = queue->Reserve(1); ASSERT_TRUE(lease.ok());
  EXPECT_FALSE(queue->Read(*lease).ok());
  EXPECT_FALSE(queue->Release(*lease).ok());
  EXPECT_FALSE(queue->Publish(*lease, Publication(1)).ok());
  EXPECT_TRUE(queue->AbortPrepared(*lease).ok());
  EXPECT_FALSE(queue->MarkInFlight(*lease).ok());
  EXPECT_TRUE(queue->Reserve(2).ok());
}
TEST(NativeV41Output, PublishedOutputRequiresConsumerReleaseBeforeSlotReuse) {
  auto queue = Queue(); ASSERT_NE(queue, nullptr);
  auto lease = queue->Reserve(1); ASSERT_TRUE(lease.ok());
  ASSERT_TRUE(queue->MarkInFlight(*lease).ok());
  ASSERT_TRUE(queue->Publish(*lease, Publication(1)).ok());
  EXPECT_FALSE(queue->Reserve(2).ok());
  EXPECT_FALSE(queue->DiscardRetired(*lease).ok());
  auto read = queue->Read(*lease); ASSERT_TRUE(read.ok());
  EXPECT_EQ(read->visible_size, 1U); EXPECT_EQ(read->visible[0], 'x');
  ASSERT_TRUE(queue->Release(*lease).ok());
  EXPECT_FALSE(queue->Read(*lease).ok());
  EXPECT_FALSE(queue->Release(*lease).ok());
  auto replacement = queue->Reserve(2); ASSERT_TRUE(replacement.ok());
  ASSERT_TRUE(queue->MarkInFlight(*replacement).ok());
  ASSERT_TRUE(queue->Publish(*replacement, Publication(2)).ok());
  EXPECT_FALSE(queue->Read(*lease).ok());
  EXPECT_FALSE(queue->Release(*lease).ok());
  EXPECT_TRUE(queue->Release(*replacement).ok());
}
TEST(NativeV41Output, RetiredInFlightReservationCanBeDiscardedButNotPublishedAgain) {
  auto queue = Queue(); ASSERT_NE(queue, nullptr);
  auto lease = queue->Reserve(7); ASSERT_TRUE(lease.ok());
  ASSERT_TRUE(queue->MarkInFlight(*lease).ok());
  EXPECT_FALSE(queue->AbortPrepared(*lease).ok());
  EXPECT_FALSE(queue->Release(*lease).ok());
  // The queue has no process custody; its caller must prove rank retirement.
  ASSERT_TRUE(queue->DiscardRetired(*lease).ok());
  EXPECT_FALSE(queue->DiscardRetired(*lease).ok());
  EXPECT_FALSE(queue->Publish(*lease, Publication(7)).ok());
  EXPECT_TRUE(queue->Reserve(8).ok());
}
TEST(NativeV41Output, ForeignQueueAndInvalidPublicationDoNotConsumeReservation) {
  auto queue = Queue(); auto other = Queue();
  ASSERT_NE(queue, nullptr); ASSERT_NE(other, nullptr);
  auto lease = queue->Reserve(3); ASSERT_TRUE(lease.ok());
  ASSERT_TRUE(queue->MarkInFlight(*lease).ok());
  EXPECT_FALSE(other->Publish(*lease, Publication(3)).ok());
  EXPECT_FALSE(other->DiscardRetired(*lease).ok());
  EXPECT_FALSE(queue->Publish(*lease, Publication(4)).ok());
  auto oversized = Publication(3);
  oversized.visible_size = static_cast<std::uint32_t>(oversized.visible.size() + 1);
  EXPECT_FALSE(queue->Publish(*lease, oversized).ok());
  EXPECT_TRUE(queue->Publish(*lease, Publication(3)).ok());
  EXPECT_FALSE(other->Read(*lease).ok());
  EXPECT_FALSE(other->Release(*lease).ok());
  EXPECT_TRUE(queue->Release(*lease).ok());
}
TEST(NativeV41Output, EmptyVisiblePublicationStillOwnsCredit) {
  auto queue = Queue(); ASSERT_NE(queue, nullptr);
  auto lease = queue->Reserve(1); ASSERT_TRUE(lease.ok());
  ASSERT_TRUE(queue->MarkInFlight(*lease).ok());
  auto hidden = Publication(1); hidden.visible_size = 0;
  ASSERT_TRUE(queue->Publish(*lease, hidden).ok());
  EXPECT_FALSE(queue->Reserve(2).ok());
  auto read = queue->Read(*lease); ASSERT_TRUE(read.ok());
  EXPECT_EQ(read->visible_size, 0U);
  EXPECT_EQ(read->record.accepted_count, 1U);
  EXPECT_TRUE(queue->Release(*lease).ok());
  EXPECT_TRUE(queue->Reserve(2).ok());
}
}  // namespace
}  // namespace pih::deepseek_v41
