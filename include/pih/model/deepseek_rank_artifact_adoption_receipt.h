#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

#include "pih/model/deepseek_rank_artifact_transfer_receiver.h"

namespace pih {

class DeepSeekRankArtifactMetadataReceiver;
class DeepSeekRankArtifactMappingOwner;

inline constexpr std::string_view kDeepSeekRankArtifactAdoptionReceiptAbi =
    "pih_deepseek_rank_artifact_adoption_receipt_v1";

Result<Sha256Digest> compile_deepseek_rank_artifact_adoption_receipt_root(
    const DeepSeekRankArtifactTransferManifest& transfer_manifest,
    const Sha256Digest& descriptor_transaction_root,
    const Sha256Digest& adopted_descriptor_set_root);

// One-way worker-side boundary. Creation consumes a completed receiver only
// after its final ACK was accepted and its SCM_RIGHTS ledger lease released.
// Descriptors remain private until a later rank mapping owner consumes this
// receipt together with the canonical mapping/tensor manifest.
class DeepSeekRankArtifactAdoptionReceipt final {
 public:
  static Result<DeepSeekRankArtifactAdoptionReceipt> Create(
      DeepSeekRankArtifactTransferReceiver receiver);

  DeepSeekRankArtifactAdoptionReceipt(
      const DeepSeekRankArtifactAdoptionReceipt&) = delete;
  DeepSeekRankArtifactAdoptionReceipt& operator=(
      const DeepSeekRankArtifactAdoptionReceipt&) = delete;
  DeepSeekRankArtifactAdoptionReceipt(
      DeepSeekRankArtifactAdoptionReceipt&&) noexcept = default;
  DeepSeekRankArtifactAdoptionReceipt& operator=(
      DeepSeekRankArtifactAdoptionReceipt&&) noexcept = default;

  [[nodiscard]] const DeepSeekRankArtifactTransferManifest&
  transfer_manifest() const noexcept {
    return transfer_manifest_;
  }
  [[nodiscard]] std::size_t descriptor_count() const noexcept {
    return descriptors_.size();
  }
  [[nodiscard]] bool retains_descriptor_owners() const noexcept {
    return !descriptors_.empty() &&
           descriptors_.size() == expectations_.size();
  }
  [[nodiscard]] const Sha256Digest& transaction_root() const noexcept {
    return transaction_root_;
  }
  [[nodiscard]] const Sha256Digest& adopted_descriptor_set_root()
      const noexcept {
    return adopted_descriptor_set_root_;
  }
  [[nodiscard]] const Sha256Digest& receipt_root() const noexcept {
    return receipt_root_;
  }

 private:
  friend class DeepSeekRankArtifactMetadataReceiver;
  friend class DeepSeekRankArtifactMappingOwner;

  DeepSeekRankArtifactAdoptionReceipt(
      DeepSeekRankArtifactTransferManifest transfer_manifest,
      std::vector<DeepSeekRankArtifactDescriptorExpectation> expectations,
      std::vector<DeepSeekWorkerShardDescriptor> descriptors,
      std::int32_t control_fd,
      Sha256Digest transaction_root,
      Sha256Digest adopted_descriptor_set_root,
      Sha256Digest receipt_root) noexcept;

  DeepSeekRankArtifactTransferManifest transfer_manifest_;
  std::vector<DeepSeekRankArtifactDescriptorExpectation> expectations_;
  std::vector<DeepSeekWorkerShardDescriptor> descriptors_;
  std::int32_t control_fd_ = -1;
  Sha256Digest transaction_root_{};
  Sha256Digest adopted_descriptor_set_root_{};
  Sha256Digest receipt_root_{};
};

}  // namespace pih
