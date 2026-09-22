#pragma once
#include "block_sequence.h"
#include "token_embedding.h"
#include "token_input_upload.h"
#include "uploaded_bindings.h"

namespace pih::deepseek_v41 {
enum class EmbeddingState { kWaitingReduction, kWaitingCompletion, kComplete, kFailed };
class EmbeddingOperation final {
 public:
  using Clock = std::chrono::steady_clock;
  EmbeddingOperation(const EmbeddingOperation&) = delete;
  EmbeddingOperation& operator=(const EmbeddingOperation&) = delete;
  EmbeddingOperation(EmbeddingOperation&& other) noexcept;
  EmbeddingOperation& operator=(EmbeddingOperation&&) = delete;
  ~EmbeddingOperation();
  static Result<EmbeddingOperation> Start(BlockSequence& sequence, TokenEmbeddingLaunch launch, const BackboneWeightUpload& weights,
      EngramHashState& hashes, std::span<const std::uint32_t> tokens, const TokenInputUploadLaunch& upload,
      std::uintptr_t communicator, const EngramCompletionResources& resources, Clock::time_point deadline);
  Result<EmbeddingState> Advance();
 private:
  EmbeddingOperation() = default;
  Result<EmbeddingState> AdvanceImpl();
  BlockSequence* sequence_ = nullptr;
  TokenEmbeddingLaunch launch_{};
  TokenInputUploadLaunch upload_{};
  std::uintptr_t communicator_ = 0;
  EngramCompletionResources resources_{};
  Clock::time_point deadline_{};
  std::optional<EngramReduction> reduction_;
  std::optional<EngramCompletion> completion_;
  EmbeddingState state_ = EmbeddingState::kFailed;
};
}  // namespace pih::deepseek_v41
