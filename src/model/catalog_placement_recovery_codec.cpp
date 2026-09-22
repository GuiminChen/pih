#include "pih/model/catalog_placement_recovery_codec.h"

#include <array>
#include <cstring>

namespace pih {
namespace {
constexpr std::string_view kDomain = "pih:catalog-placement-recovery:v1";
constexpr std::size_t kDigestCount = 6;
constexpr std::size_t kPayloadBytes = kDomain.size() + 1 + 1 + 1 + 8 + 5 * 8 + kDigestCount * 32;
constexpr std::size_t kBytes = kPayloadBytes + 32;

bool nonzero(const Sha256Digest& value) {
  std::byte aggregate{};
  for (auto byte : value.bytes) aggregate |= byte;
  return aggregate != std::byte{};
}
bool valid(const CatalogPlacementRecoverySnapshot& value) {
  const bool route_state = value.state == CatalogPlacementState::kRouteCommitting ||
                           value.state == CatalogPlacementState::kRouted ||
                           value.state == CatalogPlacementState::kRouteWithdrawing ||
                           value.state == CatalogPlacementState::kWithdrawn;
  const bool early_state = value.state == CatalogPlacementState::kVerifying ||
                           value.state == CatalogPlacementState::kStarting;
  return value.deployment_generation != 0 &&
         nonzero(value.transaction_id) && nonzero(value.target_topology_root) &&
         nonzero(value.authority_snapshot_root) && nonzero(value.bootstrap_manifest_root) &&
         nonzero(value.readiness_nonce_digest) &&
         value.absolute_deadlines[0] != 0 &&
         value.absolute_deadlines[0] <= value.absolute_deadlines[1] &&
         value.absolute_deadlines[1] <= value.absolute_deadlines[2] &&
         value.absolute_deadlines[2] <= value.absolute_deadlines[4] &&
         value.absolute_deadlines[3] <= value.absolute_deadlines[4] &&
         (!route_state || nonzero(value.readiness_receipt_root)) &&
         (!early_state || !nonzero(value.readiness_receipt_root)) &&
         static_cast<std::uint8_t>(value.state) >= 1 &&
         static_cast<std::uint8_t>(value.state) <= 10;
}
void put_u64(std::vector<std::byte>& out, std::uint64_t value) {
  for (unsigned shift = 0; shift < 64; shift += 8) out.push_back(std::byte{static_cast<unsigned char>(value >> shift)});
}
std::uint64_t get_u64(std::span<const std::byte> bytes, std::size_t& offset) {
  std::uint64_t value = 0;
  for (unsigned shift = 0; shift < 64; shift += 8) value |= static_cast<std::uint64_t>(std::to_integer<unsigned char>(bytes[offset++])) << shift;
  return value;
}
Status invalid() { return Status::InvalidArgument("catalog placement recovery encoding is invalid"); }
}  // namespace

Result<std::vector<std::byte>> encode_catalog_placement_recovery_snapshot(
    const CatalogPlacementRecoverySnapshot& value) {
  if (!valid(value)) return invalid();
  std::vector<std::byte> out;
  out.reserve(kBytes);
  const auto domain = std::as_bytes(std::span(kDomain));
  out.insert(out.end(), domain.begin(), domain.end()); out.push_back(std::byte{});
  out.push_back(std::byte{1}); out.push_back(std::byte{static_cast<unsigned char>(value.state)});
  put_u64(out, value.deployment_generation);
  for (auto deadline : value.absolute_deadlines) put_u64(out, deadline);
  for (const auto* digest : {&value.transaction_id, &value.target_topology_root, &value.authority_snapshot_root, &value.bootstrap_manifest_root, &value.readiness_nonce_digest, &value.readiness_receipt_root}) out.insert(out.end(), digest->bytes.begin(), digest->bytes.end());
  auto checksum = sha256(out);
  if (!checksum.ok()) return checksum.status();
  out.insert(out.end(), checksum->bytes.begin(), checksum->bytes.end());
  return out;
}

Result<CatalogPlacementRecoverySnapshot> parse_catalog_placement_recovery_snapshot(
    std::span<const std::byte> bytes) {
  if (bytes.size() != kBytes) return invalid();
  auto checksum = sha256(bytes.first(kPayloadBytes));
  if (!checksum.ok() || !std::equal(checksum->bytes.begin(), checksum->bytes.end(),
                                    bytes.begin() + static_cast<std::ptrdiff_t>(kPayloadBytes))) {
    return invalid();
  }
  std::size_t offset = 0;
  const auto domain = std::as_bytes(std::span(kDomain));
  if (!std::equal(domain.begin(), domain.end(), bytes.begin()) || bytes[domain.size()] != std::byte{}) return invalid();
  offset = domain.size() + 1;
  if (std::to_integer<unsigned char>(bytes[offset++]) != 1) return invalid();
  CatalogPlacementRecoverySnapshot value;
  value.state = static_cast<CatalogPlacementState>(std::to_integer<unsigned char>(bytes[offset++]));
  value.deployment_generation = get_u64(bytes, offset);
  for (auto& deadline : value.absolute_deadlines) deadline = get_u64(bytes, offset);
  for (auto* digest : {&value.transaction_id, &value.target_topology_root, &value.authority_snapshot_root, &value.bootstrap_manifest_root, &value.readiness_nonce_digest, &value.readiness_receipt_root}) {
    std::memcpy(digest->bytes.data(), bytes.data() + offset, digest->bytes.size()); offset += digest->bytes.size();
  }
  if (!valid(value)) return invalid();
  return value;
}

}  // namespace pih
