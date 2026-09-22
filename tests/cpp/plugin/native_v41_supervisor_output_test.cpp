#include "../../../plugins/model-deepseek-v41/supervisor_output.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <stdexcept>

namespace pih::deepseek_v41 {
namespace {
TokenOutputLease Publish(TokenOutputQueue& queue, unsigned count, unsigned bytes) {
  auto lease = queue.Reserve(count); EXPECT_TRUE(lease.ok());
  if (!lease.ok()) return {};
  EXPECT_TRUE(queue.MarkInFlight(*lease).ok());
  TokenPublication publication{};
  publication.record.accepted_count = count;
  publication.record.observation.identity.plan_seq = count;
  publication.visible_size = bytes;
  std::fill_n(publication.visible.begin(), bytes, 'x');
  EXPECT_TRUE(queue.Publish(*lease, publication).ok());
  return *lease;
}
struct WriterState { unsigned calls = 0; bool blocked = false; };
Result<std::size_t> Partial(void* context, std::string_view bytes) {
  auto& state = *static_cast<WriterState*>(context); ++state.calls;
  EXPECT_LE(bytes.size(), 4096U);
  return state.blocked ? 0U : std::min<std::size_t>(bytes.size(), 1024);
}
TEST(NativeV41SupervisorOutput, BackpressureAndPartialWritesRetainCredit) {
  auto queue = TokenOutputQueue::Create(1, sizeof(TokenPublication)); ASSERT_TRUE(queue.ok());
  SupervisorOutput output(**queue);
  const auto lease = Publish(**queue, 1, 5000);
  ASSERT_TRUE(output.Adopt(lease).ok());
  WriterState writer{0, true};
  auto blocked = output.Write(&writer, Partial);
  ASSERT_TRUE(blocked.ok()); EXPECT_FALSE(*blocked);
  EXPECT_TRUE(output.pending()); EXPECT_EQ(output.visible_bytes(), 0U);
  EXPECT_FALSE((*queue)->Reserve(2).ok());
  writer.blocked = false;
  for (unsigned i = 0; i < 5; ++i) ASSERT_TRUE(output.Write(&writer, Partial).ok());
  EXPECT_FALSE(output.pending()); EXPECT_EQ(output.visible_bytes(), 5000U);
  EXPECT_FALSE((*queue)->Read(lease).ok()); EXPECT_TRUE((*queue)->Reserve(2).ok());
}
TEST(NativeV41SupervisorOutput, InvalidOrderRetainsLeaseForExplicitDiscard) {
  auto queue = TokenOutputQueue::Create(1, sizeof(TokenPublication)); ASSERT_TRUE(queue.ok());
  SupervisorOutput output(**queue);
  ASSERT_FALSE(output.Adopt(Publish(**queue, 2, 1)).ok());
  EXPECT_TRUE(output.pending()); EXPECT_TRUE(output.failed());
  EXPECT_TRUE(output.Discard().ok()); EXPECT_FALSE(output.pending());
  EXPECT_TRUE((*queue)->Reserve(3).ok());
}
TEST(NativeV41SupervisorOutput, InvalidWriterCountPoisonsOutput) {
  auto queue = TokenOutputQueue::Create(1, sizeof(TokenPublication)); ASSERT_TRUE(queue.ok());
  SupervisorOutput output(**queue); ASSERT_TRUE(output.Adopt(Publish(**queue, 1, 1)).ok());
  auto result = output.Write(nullptr, [](void*, std::string_view bytes) -> Result<std::size_t> { return bytes.size() + 1; });
  EXPECT_FALSE(result.ok()); EXPECT_TRUE(output.pending()); EXPECT_TRUE(output.failed());
  WriterState writer;
  EXPECT_FALSE(output.Write(&writer, Partial).ok()); EXPECT_EQ(writer.calls, 0U);
  EXPECT_TRUE(output.Discard().ok());
}
TEST(NativeV41SupervisorOutput, EmptyPublicationReleasesWithoutCallback) {
  auto queue = TokenOutputQueue::Create(1, sizeof(TokenPublication)); ASSERT_TRUE(queue.ok());
  SupervisorOutput output(**queue); ASSERT_TRUE(output.Adopt(Publish(**queue, 1, 0)).ok());
  WriterState writer;
  auto result = output.Write(&writer, Partial);
  ASSERT_TRUE(result.ok()); EXPECT_TRUE(*result); EXPECT_EQ(writer.calls, 0U);
  EXPECT_FALSE(output.pending()); EXPECT_EQ(output.visible_bytes(), 0U);
}
TEST(NativeV41SupervisorOutput, FinishRequiresTerminalLedgerAndDeliveredPublication) {
  auto queue = TokenOutputQueue::Create(1, sizeof(TokenPublication)); ASSERT_TRUE(queue.ok());
  TokenStopConfig stopping{}; stopping.maximum = 1;
  auto ledger = TokenLedger::Create({1, 0, 1, 1}, 1, {}, stopping, sizeof(AcceptedToken), **queue);
  ASSERT_TRUE(ledger.ok());
  SupervisorOutput output(**queue);
  EXPECT_FALSE(output.Finish(**ledger).ok());
  auto sampling = (*ledger)->Prepare(1); ASSERT_TRUE(sampling.ok());
  ASSERT_TRUE((*ledger)->MarkInFlight(1).ok());
  SamplingObservation observation{};
  observation.identity = sampling->identity;
  observation.processed_length = sampling->processed_length;
  ASSERT_TRUE((*ledger)->Stage(observation, "x").ok());
  auto lease = (*ledger)->CommitLocal(1); ASSERT_TRUE(lease.ok());
  EXPECT_FALSE(output.Finish(**ledger).ok());
  ASSERT_TRUE(output.Adopt(*lease).ok());
  EXPECT_FALSE(output.Finish(**ledger).ok());
  WriterState writer;
  ASSERT_TRUE(output.Write(&writer, Partial).ok());
  EXPECT_TRUE(output.Finish(**ledger).ok());
  (*ledger)->Fail(); EXPECT_FALSE(output.Finish(**ledger).ok());
}
TEST(NativeV41SupervisorOutput, WriterExceptionRetainsCreditUntilDiscard) {
  auto queue = TokenOutputQueue::Create(1, sizeof(TokenPublication)); ASSERT_TRUE(queue.ok());
  SupervisorOutput output(**queue); ASSERT_TRUE(output.Adopt(Publish(**queue, 1, 1)).ok());
  auto result = output.Write(nullptr, [](void*, std::string_view) -> Result<std::size_t> { throw std::runtime_error("writer failure"); });
  EXPECT_FALSE(result.ok()); EXPECT_TRUE(output.failed()); EXPECT_TRUE(output.pending());
  EXPECT_TRUE(output.Discard().ok());
}
}  // namespace
}  // namespace pih::deepseek_v41
