#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string_view>

#include "pih/core/result.h"
#include "pih/core/sha256.h"
#include "pih/io/dm_verity_snapshot_receipt.h"

namespace pih {

enum class ArtifactImmutabilityMode : std::uint8_t {
  kUncalibrated = 0,
  kFsVerity = 1,
  kDmVeritySnapshot = 2,
};

struct ArtifactFileIdentity final {
  std::uint64_t file_bytes = 0;
  std::uint64_t filesystem_identity = 0;
  std::uint64_t file_identity = 0;
  std::int64_t data_mtime_seconds = 0;
  std::uint32_t data_mtime_nanoseconds = 0;
  bool operator==(const ArtifactFileIdentity&) const = default;
};

struct ArtifactDescriptorImpl;
struct ArtifactDirectoryImpl;
class DescriptorMappedRange;
class LinuxDeepSeekRankArtifactTransferControllerOperations;
class LinuxDeepSeekRankArtifactTransferReceiverOperations;

Status validate_fsverity_sha256_measurement(
    const Sha256Digest& expected, std::uint16_t measured_algorithm,
    std::span<const std::byte> measured_digest);

class ArtifactWorkerDescriptor final {
 public:
  ~ArtifactWorkerDescriptor();
  ArtifactWorkerDescriptor(const ArtifactWorkerDescriptor&) = delete;
  ArtifactWorkerDescriptor& operator=(const ArtifactWorkerDescriptor&) = delete;
  ArtifactWorkerDescriptor(ArtifactWorkerDescriptor&&) noexcept;
  ArtifactWorkerDescriptor& operator=(ArtifactWorkerDescriptor&&) noexcept;

  [[nodiscard]] const ArtifactFileIdentity& identity() const noexcept;
  Status read_exact(std::uint64_t offset, std::span<std::byte> output) const;

 private:
  explicit ArtifactWorkerDescriptor(
      std::unique_ptr<ArtifactDescriptorImpl> impl);
#ifdef __linux__
  static Result<ArtifactWorkerDescriptor> AdoptLinuxFileDescriptor(
      int* descriptor);
  [[nodiscard]] int linux_file_descriptor_for_transfer() const noexcept;
#endif
  std::unique_ptr<ArtifactDescriptorImpl> impl_;
  friend class ControllerFileLease;
  friend class DescriptorMappedRange;
  friend class LinuxDeepSeekRankArtifactTransferControllerOperations;
  friend class LinuxDeepSeekRankArtifactTransferReceiverOperations;
};

class ArtifactDirectoryAuthority final {
 public:
  static Result<ArtifactDirectoryAuthority> Open(
      const std::filesystem::path& trusted_root);

  ~ArtifactDirectoryAuthority();
  ArtifactDirectoryAuthority(const ArtifactDirectoryAuthority&) = delete;
  ArtifactDirectoryAuthority& operator=(const ArtifactDirectoryAuthority&) =
      delete;
  ArtifactDirectoryAuthority(ArtifactDirectoryAuthority&&) noexcept;
  ArtifactDirectoryAuthority& operator=(ArtifactDirectoryAuthority&&) noexcept;

 private:
  explicit ArtifactDirectoryAuthority(
      std::unique_ptr<ArtifactDirectoryImpl> impl);
  std::unique_ptr<ArtifactDirectoryImpl> impl_;
  friend class ControllerFileLease;
};

class ControllerFileLease final {
 public:
  static Result<ControllerFileLease> OpenBeneath(
      const ArtifactDirectoryAuthority& trusted_root,
      std::string_view member_basename, std::uint64_t maximum_bytes,
      ArtifactImmutabilityMode immutability_mode);
  static Result<ControllerFileLease> OpenBeneath(
      const std::filesystem::path& trusted_root,
      std::string_view member_basename, std::uint64_t maximum_bytes,
      ArtifactImmutabilityMode immutability_mode);
  static Result<ControllerFileLease> OpenBeneathFsVerity(
      const std::filesystem::path& trusted_root,
      std::string_view member_basename, std::uint64_t maximum_bytes,
      const Sha256Digest& expected_digest);
  static Result<ControllerFileLease> OpenBeneathDmVeritySnapshot(
      const std::filesystem::path& trusted_root,
      std::string_view member_basename, std::uint64_t maximum_bytes,
      const DmVeritySupervisorReceipt& supervisor_receipt,
      const Sha256Digest& expected_root_digest,
      const Sha256Digest& expected_table_digest,
      const Sha256Digest& expected_supervisor_attestation_digest,
      std::uint64_t integrity_reserve_bytes);

  ~ControllerFileLease();
  ControllerFileLease(const ControllerFileLease&) = delete;
  ControllerFileLease& operator=(const ControllerFileLease&) = delete;
  ControllerFileLease(ControllerFileLease&&) noexcept;
  ControllerFileLease& operator=(ControllerFileLease&&) noexcept;

  [[nodiscard]] const ArtifactFileIdentity& identity() const noexcept;
  [[nodiscard]] ArtifactImmutabilityMode immutability_mode() const noexcept {
    return immutability_mode_;
  }
  [[nodiscard]] bool production_eligible() const noexcept {
    return production_eligible_;
  }
  [[nodiscard]] const Sha256Digest& enforced_digest() const noexcept {
    return enforced_digest_;
  }
  [[nodiscard]] const Sha256Digest& integrity_table_digest() const noexcept {
    return integrity_table_digest_;
  }
  [[nodiscard]] std::uint64_t integrity_owner_bytes() const noexcept {
    return integrity_owner_bytes_;
  }
  Status read_exact(std::uint64_t offset, std::span<std::byte> output) const;
  Status poll_identity_unchanged() const;
  Status poll_integrity_unchanged(
      const DmVeritySupervisorReceipt* current_dm_receipt = nullptr) const;
  Result<ArtifactWorkerDescriptor> duplicate_for_worker() const;

 private:
  ControllerFileLease(std::unique_ptr<ArtifactDescriptorImpl> impl,
                      ArtifactImmutabilityMode mode,
                      bool production_eligible);

  std::unique_ptr<ArtifactDescriptorImpl> impl_;
  ArtifactImmutabilityMode immutability_mode_ =
      ArtifactImmutabilityMode::kUncalibrated;
  bool production_eligible_ = false;
  Sha256Digest enforced_digest_{};
  Sha256Digest integrity_table_digest_{};
  std::uint64_t integrity_owner_bytes_ = 0;
  Sha256Digest supervisor_attestation_digest_{};
  std::uint64_t integrity_reserve_bytes_ = 0;
};

}  // namespace pih
