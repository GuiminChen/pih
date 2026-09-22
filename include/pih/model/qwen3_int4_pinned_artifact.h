#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>

#include "pih/backend/cuda/pinned_placement_verifier.h"
#include "pih/backend/cuda/registered_pinned_allocator.h"
#include "pih/backend/cuda/typed_copy_plan.h"
#include "pih/core/buffer.h"
#include "pih/model/qwen3_int4_artifact_io.h"
namespace pih {
class QwenInt4PinnedArtifact final{
 public:
  static Result<QwenInt4PinnedArtifact> LoadVerifiedBytes(
      std::span<const std::byte> bytes, RegisteredPinnedAllocator& allocator,
      PinnedPlacementVerifier& verifier, std::int32_t numa_node,
      std::uint64_t owner_id, std::uint32_t rank,
      Sha256Digest expected_artifact_digest);
  QwenInt4PinnedArtifact(const QwenInt4PinnedArtifact&)=delete;
  QwenInt4PinnedArtifact& operator=(const QwenInt4PinnedArtifact&)=delete;
  QwenInt4PinnedArtifact(QwenInt4PinnedArtifact&&) noexcept=default;
  QwenInt4PinnedArtifact& operator=(QwenInt4PinnedArtifact&&) noexcept=default;
  [[nodiscard]] const CudaCopyEndpoint& endpoint()const noexcept{return endpoint_;}
  [[nodiscard]] const QwenInt4ArtifactVerificationReceipt& receipt()const noexcept{return receipt_;}
 private:
  QwenInt4PinnedArtifact(Buffer buffer,CudaCopyEndpoint endpoint,
      QwenInt4ArtifactVerificationReceipt receipt)
      :buffer_(std::move(buffer)),endpoint_(endpoint),receipt_(receipt){}
  Buffer buffer_;CudaCopyEndpoint endpoint_;QwenInt4ArtifactVerificationReceipt receipt_;
};
}
