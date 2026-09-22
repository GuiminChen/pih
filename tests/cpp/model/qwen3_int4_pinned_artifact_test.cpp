#include "pih/model/qwen3_int4_pinned_artifact.h"

#include <gtest/gtest.h>

namespace pih { namespace {
class NeverAllocator final:public RegisteredPinnedAllocator{
 public:Result<Allocation> allocate(std::uint64_t,std::uint64_t)override{
  ++calls;return Status::Internal("must not allocate");}
 void deallocate(Allocation)noexcept override{} int calls=0;
};
class NeverVerifier final:public PinnedPlacementVerifier{
 public:Status verify(const void*,std::uint64_t,std::int32_t)override{
  ++calls;return Status::Internal("must not verify");} int calls=0;
};
TEST(QwenInt4PinnedArtifactTest, RejectsUnauthenticatedSnapshotBeforePinnedAllocation){
 NeverAllocator allocator;NeverVerifier verifier;
 const std::byte byte{};
 auto artifact=QwenInt4PinnedArtifact::LoadVerifiedBytes(
  std::span(&byte,1),allocator,verifier,0,71,0,{});
 EXPECT_FALSE(artifact.ok());EXPECT_EQ(allocator.calls,0);EXPECT_EQ(verifier.calls,0);
}
} }
