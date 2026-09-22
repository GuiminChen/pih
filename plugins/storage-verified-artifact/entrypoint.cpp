#include "pih/plugin_sdk/abi.h"
#include "pih/contracts/verified_artifact_v1.h"
#include "pih/contracts/artifact_snapshot_v1.h"
#include "pih/core/sha256.h"
#include "pih/io/controller_file_lease.h"
#include "pih/io/descriptor_mapped_range.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <span>
#include <system_error>
#include <unordered_map>
#ifdef __linux__
#include <sys/mman.h>
#endif

namespace {

struct State final {
  struct SnapshotOwner final {
    void* address{};
    std::size_t bytes{};
    ~SnapshotOwner() {
#ifdef __linux__
      if (address != nullptr) ::munmap(address, bytes);
#endif
    }
  };
  struct MappedLease final {
    uint64_t lease_handle{};
    pih::ArtifactWorkerDescriptor descriptor;
    pih::DescriptorMappedRange mapping;
  };
  uint32_t phase{};
  const pih_host_api_v1* host{};
  std::mutex mutex;
  uint64_t next_lease_handle{1};
  uint64_t next_mapping_handle{1};
  uint64_t next_snapshot_handle{1};
  uint64_t snapshot_live_bytes{};
  bool generation_bound{};
  std::filesystem::path generation_request_root;
  std::filesystem::path generation_root;
  std::optional<pih::ArtifactDirectoryAuthority> generation_authority;
  pih::Sha256Digest generation_digest;
  uint64_t maximum_shard_bytes{};
  std::unordered_map<uint64_t, pih::ControllerFileLease> leases;
  std::unordered_map<uint64_t, std::unique_ptr<MappedLease>> mappings;
  std::unordered_map<uint64_t, std::unique_ptr<SnapshotOwner>> snapshots;
};

pih_status_v1 Status(uint32_t code, const char* message = "") {
  pih_status_v1 value{};
  value.struct_size = sizeof(value);
  value.abi_version = PIH_STATUS_ABI_VERSION_V1;
  value.code = code;
  std::strncpy(value.message, message, sizeof(value.message) - 1);
  return value;
}

pih_status_v1 CheckedHostStatus(pih_status_v1 status,
                                const char* invalid_message) {
  return pih_status_is_valid_v1(&status)
      ? status
      : Status(PIH_STATUS_INTERNAL_V1, invalid_message);
}

template <typename Operation>
pih_status_v1 ContainAbi(Operation operation, const char* failure) noexcept {
  try {
    return operation();
  } catch (const std::bad_alloc&) {
    return Status(PIH_STATUS_RESOURCE_EXHAUSTED_V1,
                  "artifact_provider_allocation_failed");
  } catch (...) {
    return Status(PIH_STATUS_INTERNAL_V1, failure);
  }
}

pih_status_v1 Advance(void* context, uint32_t expected) {
  auto& state = *static_cast<State*>(context);
  if (state.phase != expected) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "lifecycle_order_invalid");
  }
  ++state.phase;
  return Status(PIH_STATUS_OK_V1);
}

State state;

bool ValidBoundedCString(const char* value, std::size_t maximum_chars) {
  return value != nullptr && value[0] != '\0' &&
         std::memchr(value, '\0', maximum_chars + 1) != nullptr;
}

pih_status_v1 ValidateGenerationCore(
    void* context,
    const pih_verified_artifact_generation_request_v1* request) {
  if (context == nullptr || request == nullptr ||
      request->struct_size != sizeof(*request) ||
      request->abi_version != PIH_VERIFIED_ARTIFACT_ABI_VERSION_V1 ||
      !ValidBoundedCString(request->generation_root, 4096) ||
      !ValidBoundedCString(request->root_sha256_hex, 64) ||
      request->maximum_shard_bytes == 0) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "artifact_generation_request_invalid");
  }
  auto& state = *static_cast<State*>(context);
  const std::filesystem::path requested_root(request->generation_root);
  std::error_code error;
  if (!requested_root.is_absolute()) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "artifact_generation_unavailable");
  }
  auto request_identity = requested_root.lexically_normal();
  auto root = std::filesystem::canonical(requested_root, error);
  if (error || !std::filesystem::is_directory(root, error) || error) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "artifact_generation_unavailable");
  }
  const auto digest = pih::Sha256Digest::ParseHex(request->root_sha256_hex);
  if (!digest.ok()) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "artifact_generation_digest_invalid");
  }
  bool nonzero = false;
  for (const auto byte : digest->bytes) {
    nonzero = nonzero || byte != std::byte{0};
  }
  if (!nonzero) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "artifact_generation_digest_zero");
  }
  std::lock_guard lock(state.mutex);
  if (state.phase != 4) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "artifact_provider_not_ready");
  }
  if (state.generation_bound) {
    if (state.generation_request_root != request_identity ||
        state.generation_root != root || state.generation_digest != *digest ||
        state.maximum_shard_bytes != request->maximum_shard_bytes) {
      return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                    "artifact_generation_already_bound");
    }
    return Status(PIH_STATUS_OK_V1);
  }
  auto authority = pih::ArtifactDirectoryAuthority::Open(root);
  if (!authority.ok()) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "artifact_generation_authority_unavailable");
  }
  std::optional<pih::ArtifactDirectoryAuthority> completed_authority;
  completed_authority.emplace(std::move(*authority));
  state.generation_request_root.swap(request_identity);
  state.generation_root.swap(root);
  state.generation_authority.swap(completed_authority);
  state.generation_digest = *digest;
  state.maximum_shard_bytes = request->maximum_shard_bytes;
  state.generation_bound = true;
  return Status(PIH_STATUS_OK_V1);
}

bool ValidLeaseOutput(const pih_verified_artifact_lease_v1* lease) {
  return lease != nullptr && lease->struct_size == sizeof(*lease) &&
         lease->abi_version == PIH_VERIFIED_ARTIFACT_ABI_VERSION_V1 &&
         lease->handle == 0 && lease->file_bytes == 0 &&
         lease->filesystem_identity == 0 && lease->file_identity == 0 &&
         lease->data_mtime_seconds == 0 &&
         lease->data_mtime_nanoseconds == 0 && lease->reserved == 0;
}

pih_status_v1 OpenDevelopmentLeaseCore(
    void* context, const char* trusted_root, const char* member_basename,
    uint64_t maximum_bytes, pih_verified_artifact_lease_v1* lease) {
  if (context == nullptr || !ValidBoundedCString(trusted_root, 4096) ||
      !ValidBoundedCString(member_basename, 255) || maximum_bytes == 0 ||
      !ValidLeaseOutput(lease)) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "artifact_lease_request_invalid");
  }
  auto& state = *static_cast<State*>(context);
  const std::filesystem::path requested_root(trusted_root);
  const auto request_identity = requested_root.lexically_normal();
  std::lock_guard lock(state.mutex);
  if (state.phase != 4) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "artifact_provider_not_ready");
  }
  if (!requested_root.is_absolute() || !state.generation_bound ||
      request_identity != state.generation_request_root) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "artifact_lease_generation_mismatch");
  }
  if (maximum_bytes > state.maximum_shard_bytes) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "artifact_lease_limit_exceeds_generation");
  }
  if (!state.generation_authority.has_value()) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "artifact_generation_authority_missing");
  }
  auto opened = pih::ControllerFileLease::OpenBeneath(
      *state.generation_authority, member_basename, maximum_bytes,
      pih::ArtifactImmutabilityMode::kUncalibrated);
  if (!opened.ok()) return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                                  "artifact_lease_open_failed");
  if (state.next_lease_handle == 0 ||
      state.next_lease_handle == std::numeric_limits<uint64_t>::max()) {
    return Status(PIH_STATUS_RESOURCE_EXHAUSTED_V1,
                  "artifact_lease_identity_exhausted");
  }
  const auto handle = state.next_lease_handle++;
  const auto identity = opened->identity();
  auto [_, inserted] = state.leases.emplace(handle, std::move(*opened));
  if (!inserted) {
    return Status(PIH_STATUS_INTERNAL_V1,
                  "artifact_lease_identity_duplicated");
  }
  lease->handle = handle;
  lease->file_bytes = identity.file_bytes;
  lease->filesystem_identity = identity.filesystem_identity;
  lease->file_identity = identity.file_identity;
  lease->data_mtime_seconds = identity.data_mtime_seconds;
  lease->data_mtime_nanoseconds = identity.data_mtime_nanoseconds;
  lease->reserved = 0;
  return Status(PIH_STATUS_OK_V1);
}

pih_status_v1 ReadLeaseCore(void* context, uint64_t handle, uint64_t offset,
                            void* destination, uint64_t bytes) {
  if (context == nullptr || handle == 0 ||
      (bytes != 0 && destination == nullptr) ||
      bytes > std::numeric_limits<std::size_t>::max()) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "artifact_lease_read_invalid");
  }
  auto& state = *static_cast<State*>(context);
  std::lock_guard lock(state.mutex);
  if (state.phase != 4) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "artifact_provider_not_ready");
  }
  const auto found = state.leases.find(handle);
  if (found == state.leases.end()) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "artifact_lease_owner_mismatch");
  }
  auto output = std::span<std::byte>(static_cast<std::byte*>(destination),
                                     static_cast<std::size_t>(bytes));
  const auto read = found->second.read_exact(offset, output);
  return read.ok() ? Status(PIH_STATUS_OK_V1)
                   : Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                            "artifact_lease_read_failed");
}

pih_status_v1 PollLeaseCore(void* context, uint64_t handle) {
  if (context == nullptr || handle == 0) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "artifact_lease_poll_invalid");
  }
  auto& state = *static_cast<State*>(context);
  std::lock_guard lock(state.mutex);
  if (state.phase != 4) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "artifact_provider_not_ready");
  }
  const auto found = state.leases.find(handle);
  if (found == state.leases.end()) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "artifact_lease_owner_mismatch");
  }
  const auto polled = found->second.poll_identity_unchanged();
  return polled.ok() ? Status(PIH_STATUS_OK_V1)
                     : Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                              "artifact_lease_identity_changed");
}

pih_status_v1 ReleaseLeaseCore(void* context,
                               pih_verified_artifact_lease_v1* lease) {
  if (context == nullptr || lease == nullptr ||
      lease->struct_size != sizeof(*lease) ||
      lease->abi_version != PIH_VERIFIED_ARTIFACT_ABI_VERSION_V1 ||
      lease->handle == 0) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "artifact_lease_handle_invalid");
  }
  auto& state = *static_cast<State*>(context);
  std::lock_guard lock(state.mutex);
  const auto found = state.leases.find(lease->handle);
  if (found == state.leases.end()) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "artifact_lease_owner_mismatch");
  }
  const auto identity = found->second.identity();
  if (lease->file_bytes != identity.file_bytes ||
      lease->filesystem_identity != identity.filesystem_identity ||
      lease->file_identity != identity.file_identity ||
      lease->data_mtime_seconds != identity.data_mtime_seconds ||
      lease->data_mtime_nanoseconds != identity.data_mtime_nanoseconds ||
      lease->reserved != 0) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "artifact_lease_owner_mismatch");
  }
  for (const auto& [_, mapping] : state.mappings) {
    if (mapping->lease_handle == lease->handle) {
      return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                    "artifact_lease_has_live_mappings");
    }
  }
  state.leases.erase(found);
  *lease = {sizeof(*lease), PIH_VERIFIED_ARTIFACT_ABI_VERSION_V1};
  return Status(PIH_STATUS_OK_V1);
}

bool ValidMappingOutput(const pih_verified_artifact_mapping_v1* mapping) {
  return mapping != nullptr && mapping->struct_size == sizeof(*mapping) &&
         mapping->abi_version == PIH_VERIFIED_ARTIFACT_ABI_VERSION_V1 &&
         mapping->handle == 0 && mapping->lease_handle == 0 &&
         mapping->address == 0 && mapping->file_offset == 0 &&
         mapping->bytes == 0;
}

pih_status_v1 MapLeaseCore(void* context, uint64_t lease_handle,
                           uint64_t file_offset, uint64_t bytes,
                           pih_verified_artifact_mapping_v1* mapping) {
  if (context == nullptr || lease_handle == 0 || bytes == 0 ||
      !ValidMappingOutput(mapping)) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "artifact_mapping_request_invalid");
  }
  auto& state = *static_cast<State*>(context);
  std::lock_guard lock(state.mutex);
  if (state.phase != 4) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "artifact_provider_not_ready");
  }
  const auto lease = state.leases.find(lease_handle);
  if (lease == state.leases.end()) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "artifact_lease_owner_mismatch");
  }
  auto descriptor = lease->second.duplicate_for_worker();
  if (!descriptor.ok()) {
    return Status(PIH_STATUS_RESOURCE_EXHAUSTED_V1,
                  "artifact_descriptor_duplicate_failed");
  }
  auto mapped = pih::DescriptorMappedRange::MapReadOnly(
      *descriptor, file_offset, bytes);
  if (!mapped.ok()) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "artifact_mapping_failed");
  }
  if (state.next_mapping_handle == 0 ||
      state.next_mapping_handle == std::numeric_limits<uint64_t>::max()) {
    return Status(PIH_STATUS_RESOURCE_EXHAUSTED_V1,
                  "artifact_mapping_identity_exhausted");
  }
  const auto handle = state.next_mapping_handle++;
  auto owned = std::make_unique<State::MappedLease>(State::MappedLease{
      lease_handle, std::move(*descriptor), std::move(*mapped)});
  const auto address = reinterpret_cast<uintptr_t>(owned->mapping.data());
  auto [_, inserted] = state.mappings.emplace(handle, std::move(owned));
  if (!inserted) {
    return Status(PIH_STATUS_INTERNAL_V1,
                  "artifact_mapping_identity_duplicated");
  }
  mapping->handle = handle;
  mapping->lease_handle = lease_handle;
  mapping->address = address;
  mapping->file_offset = file_offset;
  mapping->bytes = bytes;
  return Status(PIH_STATUS_OK_V1);
}

pih_status_v1 ReleaseMappingCore(
    void* context, pih_verified_artifact_mapping_v1* mapping) {
  if (context == nullptr || mapping == nullptr ||
      mapping->struct_size != sizeof(*mapping) ||
      mapping->abi_version != PIH_VERIFIED_ARTIFACT_ABI_VERSION_V1 ||
      mapping->handle == 0 || mapping->lease_handle == 0 ||
      mapping->address == 0 || mapping->bytes == 0) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "artifact_mapping_handle_invalid");
  }
  auto& state = *static_cast<State*>(context);
  std::lock_guard lock(state.mutex);
  const auto found = state.mappings.find(mapping->handle);
  if (found == state.mappings.end() ||
      found->second->lease_handle != mapping->lease_handle ||
      reinterpret_cast<uintptr_t>(found->second->mapping.data()) !=
          mapping->address ||
      found->second->mapping.file_offset() != mapping->file_offset ||
      found->second->mapping.size_bytes() != mapping->bytes) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "artifact_mapping_owner_mismatch");
  }
  state.mappings.erase(found);
  *mapping = {sizeof(*mapping), PIH_VERIFIED_ARTIFACT_ABI_VERSION_V1};
  return Status(PIH_STATUS_OK_V1);
}

pih_status_v1 OpenSnapshotCore(
    void* context, const char* trusted_root, const char* member_basename,
    const char* expected_sha256_hex, uint64_t maximum_bytes,
    pih_artifact_snapshot_v1* snapshot) {
#ifndef __linux__
  return Status(PIH_STATUS_UNAVAILABLE_V1, "artifact_snapshot_requires_linux");
#else
  constexpr uint64_t kSnapshotBudget = 2ULL * 1024 * 1024 * 1024;
  if (context != &state || !ValidBoundedCString(trusted_root, 4096) ||
      !ValidBoundedCString(member_basename, 255) ||
      !ValidBoundedCString(expected_sha256_hex, 64) ||
      std::strlen(expected_sha256_hex) != 64 || maximum_bytes == 0 ||
      maximum_bytes > kSnapshotBudget ||
      maximum_bytes > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max()) ||
      !snapshot || snapshot->struct_size != sizeof(*snapshot) ||
      snapshot->abi_version != PIH_ARTIFACT_SNAPSHOT_ABI_VERSION_V1 ||
      snapshot->handle || snapshot->address || snapshot->bytes)
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1, "artifact_snapshot_request_invalid");
  auto expected = pih::Sha256Digest::ParseHex(expected_sha256_hex);
  if (!expected.ok() || *expected == pih::Sha256Digest{})
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1, "artifact_snapshot_digest_invalid");
  const std::filesystem::path root(trusted_root);
  if (!root.is_absolute())
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1, "artifact_snapshot_root_invalid");
  auto& owner = *static_cast<State*>(context);
  std::lock_guard lock(owner.mutex);
  if (owner.phase != 4)
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "artifact_provider_not_ready");
  auto lease = pih::ControllerFileLease::OpenBeneath(
      root, member_basename, maximum_bytes,
      pih::ArtifactImmutabilityMode::kUncalibrated);
  if (!lease.ok() || lease->identity().file_bytes == 0 ||
      lease->identity().file_bytes > maximum_bytes)
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "artifact_snapshot_open_failed");
  if (owner.snapshots.size() >= 4 || owner.snapshot_live_bytes > kSnapshotBudget ||
      lease->identity().file_bytes > kSnapshotBudget - owner.snapshot_live_bytes)
    return Status(PIH_STATUS_RESOURCE_EXHAUSTED_V1, "artifact_snapshot_budget_exhausted");
  const auto bytes = static_cast<std::size_t>(lease->identity().file_bytes);
  auto owned = std::make_unique<State::SnapshotOwner>();
  void* address = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (address == MAP_FAILED)
    return Status(PIH_STATUS_RESOURCE_EXHAUSTED_V1, "artifact_snapshot_allocation_failed");
  owned->address = address;
  owned->bytes = bytes;
  pih::Sha256 digest;
  constexpr std::size_t kChunk = 1U << 20;
  for (std::size_t offset = 0; offset < bytes;) {
    const auto count = std::min(kChunk, bytes - offset);
    auto part = std::span<std::byte>(static_cast<std::byte*>(address) + offset, count);
    if (!lease->read_exact(offset, part).ok() || !digest.update(part).ok())
      return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "artifact_snapshot_read_failed");
    offset += count;
  }
  auto actual = digest.finalize();
  if (!actual.ok() || *actual != *expected ||
      !lease->poll_identity_unchanged().ok())
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "artifact_snapshot_digest_mismatch");
  if (::mprotect(address, bytes, PROT_READ) != 0)
    return Status(PIH_STATUS_UNAVAILABLE_V1, "artifact_snapshot_seal_failed");
  if (owner.next_snapshot_handle == 0 ||
      owner.next_snapshot_handle == std::numeric_limits<uint64_t>::max())
    return Status(PIH_STATUS_RESOURCE_EXHAUSTED_V1, "artifact_snapshot_identity_exhausted");
  const auto handle = owner.next_snapshot_handle++;
  auto [_, inserted] = owner.snapshots.emplace(handle, std::move(owned));
  if (!inserted)
    return Status(PIH_STATUS_INTERNAL_V1, "artifact_snapshot_identity_duplicated");
  owner.snapshot_live_bytes += bytes;
  *snapshot = {sizeof(*snapshot), PIH_ARTIFACT_SNAPSHOT_ABI_VERSION_V1,
               handle, reinterpret_cast<uintptr_t>(address), bytes};
  return Status(PIH_STATUS_OK_V1);
#endif
}

pih_status_v1 ReleaseSnapshotCore(void* context,
                                  pih_artifact_snapshot_v1* snapshot) {
  if (context != &state || !snapshot || snapshot->struct_size != sizeof(*snapshot) ||
      snapshot->abi_version != PIH_ARTIFACT_SNAPSHOT_ABI_VERSION_V1 ||
      !snapshot->handle || !snapshot->address || !snapshot->bytes)
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1, "artifact_snapshot_release_invalid");
  auto& owner = *static_cast<State*>(context);
  std::lock_guard lock(owner.mutex);
  auto found = owner.snapshots.find(snapshot->handle);
  if (found == owner.snapshots.end() ||
      reinterpret_cast<uintptr_t>(found->second->address) != snapshot->address ||
      found->second->bytes != snapshot->bytes)
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "artifact_snapshot_owner_mismatch");
  if (owner.snapshot_live_bytes < snapshot->bytes)
    return Status(PIH_STATUS_INTERNAL_V1, "artifact_snapshot_budget_corrupted");
  owner.snapshot_live_bytes -= snapshot->bytes;
  owner.snapshots.erase(found);
  *snapshot = {sizeof(*snapshot), PIH_ARTIFACT_SNAPSHOT_ABI_VERSION_V1};
  return Status(PIH_STATUS_OK_V1);
}

pih_status_v1 OpenSnapshot(void* context, const char* root, const char* name,
                           const char* digest, uint64_t maximum,
                           pih_artifact_snapshot_v1* snapshot) noexcept {
  return ContainAbi([&] { return OpenSnapshotCore(context, root, name, digest,
                                                  maximum, snapshot); },
                    "artifact_snapshot_open_failed");
}
pih_status_v1 ReleaseSnapshot(void* context,
                              pih_artifact_snapshot_v1* snapshot) noexcept {
  return ContainAbi([&] { return ReleaseSnapshotCore(context, snapshot); },
                    "artifact_snapshot_release_failed");
}
pih_artifact_snapshot_api_v1 snapshot_api{
    sizeof(snapshot_api), PIH_ARTIFACT_SNAPSHOT_ABI_VERSION_V1,
    &state, OpenSnapshot, ReleaseSnapshot};

pih_status_v1 ValidateGeneration(
    void* context,
    const pih_verified_artifact_generation_request_v1* request) noexcept {
  return ContainAbi(
      [&] { return ValidateGenerationCore(context, request); },
      "artifact_generation_validation_failed");
}

pih_status_v1 OpenDevelopmentLease(
    void* context, const char* trusted_root, const char* member_basename,
    uint64_t maximum_bytes, pih_verified_artifact_lease_v1* lease) noexcept {
  return ContainAbi(
      [&] {
        return OpenDevelopmentLeaseCore(context, trusted_root,
                                        member_basename, maximum_bytes, lease);
      },
      "artifact_lease_open_failed");
}

pih_status_v1 ReadLease(void* context, uint64_t handle, uint64_t offset,
                        void* destination, uint64_t bytes) noexcept {
  return ContainAbi(
      [&] { return ReadLeaseCore(context, handle, offset, destination, bytes); },
      "artifact_lease_read_failed");
}

pih_status_v1 PollLease(void* context, uint64_t handle) noexcept {
  return ContainAbi([&] { return PollLeaseCore(context, handle); },
                    "artifact_lease_poll_failed");
}

pih_status_v1 ReleaseLease(
    void* context, pih_verified_artifact_lease_v1* lease) noexcept {
  return ContainAbi([&] { return ReleaseLeaseCore(context, lease); },
                    "artifact_lease_release_failed");
}

pih_status_v1 MapLease(
    void* context, uint64_t lease_handle, uint64_t file_offset, uint64_t bytes,
    pih_verified_artifact_mapping_v1* mapping) noexcept {
  return ContainAbi(
      [&] {
        return MapLeaseCore(context, lease_handle, file_offset, bytes,
                            mapping);
      },
      "artifact_mapping_failed");
}

pih_status_v1 ReleaseMapping(
    void* context, pih_verified_artifact_mapping_v1* mapping) noexcept {
  return ContainAbi([&] { return ReleaseMappingCore(context, mapping); },
                    "artifact_mapping_release_failed");
}

pih_verified_artifact_api_v1 artifact_api{
    sizeof(pih_verified_artifact_api_v1), PIH_VERIFIED_ARTIFACT_ABI_VERSION_V1,
    &state, &ValidateGeneration, &OpenDevelopmentLease, &ReadLease, &PollLease,
    &ReleaseLease, &MapLease, &ReleaseMapping};

pih_status_v1 RegisterCore(void* context) {
  auto& state = *static_cast<State*>(context);
  if (state.host->register_capability == nullptr) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "capability_registry_missing");
  }
  pih_capability_v1 capability{};
  capability.struct_size = sizeof(capability);
  capability.abi_version = PIH_CAPABILITY_ABI_VERSION_V1;
  capability.capability_id = "artifact.verified-reader.v1";
  capability.contract_id = "pih.artifact.verified-reader.v1";
  capability.api = &artifact_api;
  capability.threading_model = PIH_CAPABILITY_THREADING_SERIALIZED_V1;
  capability.scope = PIH_CAPABILITY_SCOPE_ACTIVATION_V1;
  capability.cardinality = PIH_CAPABILITY_CARDINALITY_EXACTLY_ONE_V1;
  const auto status = CheckedHostStatus(
      state.host->register_capability(state.host->context, &capability),
      "capability_registry_status_invalid");
  if (!pih_status_is_ok_v1(&status)) return status;
  pih_capability_v1 snapshot_capability{};
  snapshot_capability.struct_size = sizeof(snapshot_capability);
  snapshot_capability.abi_version = PIH_CAPABILITY_ABI_VERSION_V1;
  snapshot_capability.capability_id = "artifact.authenticated-snapshot.v1";
  snapshot_capability.contract_id = "pih.artifact.authenticated-snapshot.v1";
  snapshot_capability.api = &snapshot_api;
  snapshot_capability.threading_model = PIH_CAPABILITY_THREADING_SERIALIZED_V1;
  snapshot_capability.scope = PIH_CAPABILITY_SCOPE_ACTIVATION_V1;
  snapshot_capability.cardinality = PIH_CAPABILITY_CARDINALITY_EXACTLY_ONE_V1;
  const auto snapshot_status = CheckedHostStatus(
      state.host->register_capability(state.host->context, &snapshot_capability),
      "snapshot_registry_status_invalid");
  if (!pih_status_is_ok_v1(&snapshot_status)) return snapshot_status;
  return Advance(context, 0);
}
pih_status_v1 ConfigureCore(void* context) { return Advance(context, 1); }
pih_status_v1 StartCore(void* context) { return Advance(context, 2); }
pih_status_v1 ReadyCore(void* context) { return Advance(context, 3); }
pih_status_v1 DrainCore(void* context) {
  auto& state = *static_cast<State*>(context);
  std::lock_guard lock(state.mutex);
  if (!state.mappings.empty() || !state.leases.empty() || !state.snapshots.empty()) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "artifact_resources_still_live");
  }
  return Advance(context, 4);
}
pih_status_v1 StopCore(void* context) {
  auto& state = *static_cast<State*>(context);
  std::lock_guard lock(state.mutex);
  if (state.phase == 3) {
    state.phase = 6;
    return Status(PIH_STATUS_OK_V1);
  }
  return Advance(context, 5);
}
pih_status_v1 DisposeCore(void* context) {
  auto& state = *static_cast<State*>(context);
  {
    std::lock_guard lock(state.mutex);
    if (state.phase != 1 && state.phase != 2 && state.phase != 6) {
      return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                    "lifecycle_order_invalid");
    }
    if (!state.mappings.empty() || !state.leases.empty() || !state.snapshots.empty()) {
      return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                    "artifact_resources_still_live");
    }
    state.generation_bound = false;
    state.generation_request_root.clear();
    state.generation_root.clear();
    state.generation_authority.reset();
    state.generation_digest = {};
    state.maximum_shard_bytes = 0;
    state.host = nullptr;
    state.phase = 7;
  }
  return Status(PIH_STATUS_OK_V1);
}

pih_status_v1 ContainLifecycle(void* context,
                               pih_lifecycle_callback_v1 callback,
                               const char* failure) noexcept {
  if (context == nullptr || callback == nullptr) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "artifact_lifecycle_context_invalid");
  }
  return ContainAbi([&] { return callback(context); }, failure);
}

pih_status_v1 Register(void* context) noexcept {
  return ContainLifecycle(context, &RegisterCore,
                          "artifact_registration_failed");
}
pih_status_v1 Configure(void* context) noexcept {
  return ContainLifecycle(context, &ConfigureCore,
                          "artifact_configuration_failed");
}
pih_status_v1 Start(void* context) noexcept {
  return ContainLifecycle(context, &StartCore, "artifact_start_failed");
}
pih_status_v1 Ready(void* context) noexcept {
  return ContainLifecycle(context, &ReadyCore, "artifact_ready_failed");
}
pih_status_v1 Drain(void* context) noexcept {
  return ContainLifecycle(context, &DrainCore, "artifact_drain_failed");
}
pih_status_v1 Stop(void* context) noexcept {
  return ContainLifecycle(context, &StopCore, "artifact_stop_failed");
}
pih_status_v1 Dispose(void* context) noexcept {
  return ContainLifecycle(context, &DisposeCore,
                          "artifact_dispose_failed");
}

}  // namespace

extern "C" PIH_PLUGIN_EXPORT pih_status_v1
pih_plugin_entry_v1(const pih_host_api_v1* host,
                    pih_plugin_api_v1* plugin) noexcept {
  if (!pih_host_api_is_valid_v1(host)) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1, "host_api_invalid");
  }
  if (!pih_plugin_api_accepts_v1(plugin)) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1, "plugin_api_invalid");
  }
  if (state.host != nullptr || state.phase != 0) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "plugin_entry_already_bound");
  }
  plugin->abi_version = PIH_PLUGIN_ABI_VERSION_V1;
  plugin->plugin_id = "pih.storage.verified-artifact";
  plugin->plugin_version = "1.0.0";
  state.host = host;
  plugin->context = &state;
  plugin->lifecycle.struct_size = sizeof(pih_plugin_lifecycle_v1);
  plugin->lifecycle.abi_version = PIH_PLUGIN_LIFECYCLE_ABI_VERSION_V1;
  plugin->lifecycle.register_plugin = &Register;
  plugin->lifecycle.configure = &Configure;
  plugin->lifecycle.start = &Start;
  plugin->lifecycle.ready = &Ready;
  plugin->lifecycle.drain = &Drain;
  plugin->lifecycle.stop = &Stop;
  plugin->lifecycle.dispose = &Dispose;
  return Status(PIH_STATUS_OK_V1);
}
