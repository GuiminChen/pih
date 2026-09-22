#include "pih/model/deepseek_verified_artifact_lease.h"

#include <algorithm>
#include <array>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "pih/core/checked_math.h"
#include "pih/core/sha256.h"
#include "pih/plugin_sdk/status_bridge.h"

namespace pih {

DeepSeekVerifiedArtifactMapping::DeepSeekVerifiedArtifactMapping(
    DeepSeekVerifiedArtifactMapping&& other) noexcept
    : api_(std::exchange(other.api_, nullptr)),
      mapping_(std::exchange(other.mapping_, {})) {}

DeepSeekVerifiedArtifactMapping&
DeepSeekVerifiedArtifactMapping::operator=(
    DeepSeekVerifiedArtifactMapping&& other) noexcept {
  if (this != &other) {
    release();
    api_ = std::exchange(other.api_, nullptr);
    mapping_ = std::exchange(other.mapping_, {});
  }
  return *this;
}

DeepSeekVerifiedArtifactMapping::~DeepSeekVerifiedArtifactMapping() {
  release();
}

std::span<const std::byte> DeepSeekVerifiedArtifactMapping::bytes()
    const noexcept {
  return {reinterpret_cast<const std::byte*>(mapping_.address),
          static_cast<std::size_t>(mapping_.bytes)};
}

Result<std::span<const std::byte>>
DeepSeekVerifiedArtifactMapping::slice(std::uint64_t offset,
                                       std::uint64_t bytes) const {
  auto end = checked_add_u64(offset, bytes);
  if (!end.ok() || bytes == 0 || *end > mapping_.bytes ||
      bytes > std::numeric_limits<std::size_t>::max()) {
    return Status::InvalidArgument(
        "verified artifact mapping slice is out of bounds");
  }
  return this->bytes().subspan(static_cast<std::size_t>(offset),
                               static_cast<std::size_t>(bytes));
}

void DeepSeekVerifiedArtifactMapping::release() noexcept {
  if (api_ != nullptr && mapping_.handle != 0) {
    (void)api_->release_mapping(api_->context, &mapping_);
  }
  api_ = nullptr;
  mapping_ = {};
}

Result<DeepSeekVerifiedArtifactLease>
DeepSeekVerifiedArtifactLease::OpenDevelopment(
    const pih_verified_artifact_api_v1& api,
    const std::filesystem::path& trusted_root,
    std::string_view member_basename, std::uint64_t maximum_bytes) {
  if (api.struct_size != sizeof(api) ||
      api.contract_version != PIH_VERIFIED_ARTIFACT_ABI_VERSION_V1 ||
      api.context == nullptr || api.open_development_lease == nullptr ||
      api.read_lease == nullptr || api.poll_lease == nullptr ||
      api.release_lease == nullptr || api.map_lease == nullptr ||
      api.release_mapping == nullptr || trusted_root.empty() ||
      member_basename.empty() || member_basename.size() > 255 ||
      maximum_bytes == 0) {
    return Status::InvalidArgument("verified artifact lease input is invalid");
  }
  const auto root = trusted_root.string();
  const std::string member(member_basename);
  pih_verified_artifact_lease_v1 lease{};
  lease.struct_size = sizeof(lease);
  lease.abi_version = PIH_VERIFIED_ARTIFACT_ABI_VERSION_V1;
  auto status = plugin_status(api.open_development_lease(
      api.context, root.c_str(), member.c_str(), maximum_bytes, &lease));
  if (!status.ok()) return status;
  if (lease.struct_size != sizeof(lease) ||
      lease.abi_version != PIH_VERIFIED_ARTIFACT_ABI_VERSION_V1 ||
      lease.handle == 0 || lease.file_bytes > maximum_bytes ||
      lease.data_mtime_nanoseconds >= 1'000'000'000U ||
      lease.reserved != 0) {
    (void)api.release_lease(api.context, &lease);
    return Status::FailedPrecondition(
        "verified artifact provider returned an invalid lease");
  }
  return DeepSeekVerifiedArtifactLease(api, lease);
}

DeepSeekVerifiedArtifactLease::DeepSeekVerifiedArtifactLease(
    DeepSeekVerifiedArtifactLease&& other) noexcept
    : api_(std::exchange(other.api_, nullptr)),
      lease_(std::exchange(other.lease_, {})) {}

DeepSeekVerifiedArtifactLease& DeepSeekVerifiedArtifactLease::operator=(
    DeepSeekVerifiedArtifactLease&& other) noexcept {
  if (this != &other) {
    release();
    api_ = std::exchange(other.api_, nullptr);
    lease_ = std::exchange(other.lease_, {});
  }
  return *this;
}

DeepSeekVerifiedArtifactLease::~DeepSeekVerifiedArtifactLease() { release(); }

Status DeepSeekVerifiedArtifactLease::read_exact(
    std::uint64_t offset, std::span<std::byte> output) const {
  if (api_ == nullptr || lease_.handle == 0) {
    return Status::FailedPrecondition("verified artifact lease is closed");
  }
  return plugin_status(api_->read_lease(api_->context, lease_.handle, offset,
                                        output.data(), output.size()));
}

Status DeepSeekVerifiedArtifactLease::poll_identity_unchanged() const {
  if (api_ == nullptr || lease_.handle == 0) {
    return Status::FailedPrecondition("verified artifact lease is closed");
  }
  return plugin_status(api_->poll_lease(api_->context, lease_.handle));
}

Result<DeepSeekVerifiedArtifactMapping>
DeepSeekVerifiedArtifactLease::map_read_only(
    std::uint64_t file_offset, std::uint64_t bytes) const {
  if (api_ == nullptr || lease_.handle == 0 || bytes == 0 ||
      bytes > std::numeric_limits<std::size_t>::max()) {
    return Status::InvalidArgument("verified artifact mapping is invalid");
  }
  pih_verified_artifact_mapping_v1 mapping{};
  mapping.struct_size = sizeof(mapping);
  mapping.abi_version = PIH_VERIFIED_ARTIFACT_ABI_VERSION_V1;
  auto status = plugin_status(api_->map_lease(
      api_->context, lease_.handle, file_offset, bytes, &mapping));
  if (!status.ok()) return status;
  if (mapping.struct_size != sizeof(mapping) ||
      mapping.abi_version != PIH_VERIFIED_ARTIFACT_ABI_VERSION_V1 ||
      mapping.handle == 0 || mapping.lease_handle != lease_.handle ||
      mapping.address == 0 || mapping.file_offset != file_offset ||
      mapping.bytes != bytes) {
    (void)api_->release_mapping(api_->context, &mapping);
    return Status::FailedPrecondition(
        "verified artifact provider returned an invalid mapping");
  }
  return DeepSeekVerifiedArtifactMapping(*api_, mapping);
}

void DeepSeekVerifiedArtifactLease::release() noexcept {
  if (api_ != nullptr && lease_.handle != 0) {
    (void)api_->release_lease(api_->context, &lease_);
  }
  api_ = nullptr;
  lease_ = {};
}

Result<SafetensorsHeaderFileReceipt> load_safetensors_header_capability(
    const DeepSeekVerifiedArtifactLease& lease) {
  const auto file_bytes = lease.file_bytes();
  if (file_bytes < 8) {
    return Status::InvalidArgument(
        "Safetensors capability is shorter than length prefix");
  }
  std::array<std::byte, 8> length_prefix{};
  auto status = lease.read_exact(0, length_prefix);
  if (!status.ok()) return status;
  std::uint64_t header_bytes = 0;
  for (std::size_t index = 0; index < length_prefix.size(); ++index) {
    header_bytes |= static_cast<std::uint64_t>(
        std::to_integer<unsigned char>(length_prefix[index])) << (index * 8U);
  }
  if (header_bytes == 0 || header_bytes > SafetensorsHeader::kMaxHeaderBytes ||
      header_bytes > file_bytes - 8) {
    return Status::ResourceExhausted(
        "Safetensors capability header length is outside budget");
  }
  std::vector<std::byte> prefix(static_cast<std::size_t>(header_bytes + 8));
  std::copy(length_prefix.begin(), length_prefix.end(), prefix.begin());
  status = lease.read_exact(8, std::span<std::byte>(prefix).subspan(8));
  if (!status.ok()) return status;
  auto header = SafetensorsHeader::ParsePrefix(prefix, file_bytes);
  if (!header.ok()) return header.status();
  auto digest = sha256(prefix);
  if (!digest.ok()) return digest.status();
  return SafetensorsHeaderFileReceipt{
      std::move(*header), file_bytes, header_bytes + 8, *digest};
}

}  // namespace pih
