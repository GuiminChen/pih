#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "pih/model/deepseek_rank_artifact_metadata_receiver.h"

namespace pih {

class DeepSeekRankArtifactMappingOwner;

inline constexpr std::string_view kDeepSeekRankArtifactMetadataReceiptAbi =
    "pih_deepseek_rank_artifact_metadata_receipt_v1";

Result<Sha256Digest> compile_deepseek_rank_artifact_metadata_receipt_root(
    const DeepSeekRankArtifactTransferManifest& transfer_manifest,
    const Sha256Digest& adoption_receipt_root,
    const Sha256Digest& descriptor_transaction_root,
    const Sha256Digest& metadata_transaction_root,
    const Sha256Digest& metadata_root,
    const Sha256Digest& blob_sha256, std::uint64_t blob_bytes,
    bool dspark_enabled);

// One-way join after the final metadata ACK. Descriptors, mapping intervals
// and tensor records remain private for a later mapping owner.
class DeepSeekRankArtifactMetadataReceipt final {
 public:
  static Result<DeepSeekRankArtifactMetadataReceipt> Create(
      DeepSeekRankArtifactMetadataReceiver receiver);

  DeepSeekRankArtifactMetadataReceipt(
      const DeepSeekRankArtifactMetadataReceipt&) = delete;
  DeepSeekRankArtifactMetadataReceipt& operator=(
      const DeepSeekRankArtifactMetadataReceipt&) = delete;
  DeepSeekRankArtifactMetadataReceipt(
      DeepSeekRankArtifactMetadataReceipt&&) noexcept = default;
  DeepSeekRankArtifactMetadataReceipt& operator=(
      DeepSeekRankArtifactMetadataReceipt&&) noexcept = default;

  [[nodiscard]] bool retains_descriptor_and_metadata_owners()
      const noexcept {
    return adoption_receipt_.retains_descriptor_owners() &&
           blob_.metadata_root() != Sha256Digest{};
  }
  [[nodiscard]] const Sha256Digest& descriptor_transaction_root()
      const noexcept {
    return adoption_receipt_.transaction_root();
  }
  [[nodiscard]] const Sha256Digest& metadata_transaction_root()
      const noexcept {
    return metadata_transaction_root_;
  }
  [[nodiscard]] const Sha256Digest& metadata_root() const noexcept {
    return blob_.metadata_root();
  }
  [[nodiscard]] const Sha256Digest& blob_sha256() const noexcept {
    return blob_sha256_;
  }
  [[nodiscard]] std::uint64_t blob_bytes() const noexcept {
    return blob_bytes_;
  }
  [[nodiscard]] const Sha256Digest& receipt_root() const noexcept {
    return receipt_root_;
  }

 private:
  friend class DeepSeekRankArtifactMappingOwner;

  DeepSeekRankArtifactMetadataReceipt(
      DeepSeekRankArtifactAdoptionReceipt adoption_receipt,
      DeepSeekRankArtifactMetadataBlob blob,
      Sha256Digest metadata_transaction_root,
      Sha256Digest blob_sha256, std::uint64_t blob_bytes,
      Sha256Digest receipt_root) noexcept;

  DeepSeekRankArtifactAdoptionReceipt adoption_receipt_;
  DeepSeekRankArtifactMetadataBlob blob_;
  Sha256Digest metadata_transaction_root_{};
  Sha256Digest blob_sha256_{};
  std::uint64_t blob_bytes_ = 0;
  Sha256Digest receipt_root_{};
};

}  // namespace pih
