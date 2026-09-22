#include "pih/model/deepseek_nccl_bootstrap_capability_issuer.h"

#include "pih/core/checked_math.h"

namespace pih {
namespace {

void secure_zero(auto& bytes) noexcept {
  volatile std::byte* output = bytes.data();
  for (std::size_t i = 0; i < bytes.size(); ++i) output[i] = std::byte{0};
}

std::uint64_t commitment_id(const Sha256Digest& digest) noexcept {
  std::uint64_t value = 0;
  for (unsigned i = 0; i < 8; ++i) {
    value |= std::uint64_t{std::to_integer<std::uint8_t>(digest.bytes[i])}
             << (8U * i);
  }
  return value;
}

}  // namespace

Result<std::vector<DeepSeekNcclIssuedEdgeCapability>>
DeepSeekNcclBootstrapCapabilityIssuer::Issue(
    std::uint64_t engine_epoch, std::uint32_t world_size,
    std::uint64_t first_lease_id, DeepSeekNcclUniqueIdSource& source,
    RegisteredPinnedAllocator& allocator) {
  if (engine_epoch == 0 || world_size < 2 || world_size > 4 ||
      first_lease_id == 0) {
    return Status::InvalidArgument(
        "DeepSeek NCCL bootstrap issuance identity is invalid");
  }
  auto final_lease = checked_add_u64(first_lease_id, world_size - 2);
  if (!final_lease.ok()) return final_lease.status();
  std::vector<DeepSeekNcclIssuedEdgeCapability> issued;
  issued.reserve(world_size - 1);
  for (std::uint32_t edge = 0; edge + 1 < world_size; ++edge) {
    auto raw = source.get_unique_id();
    if (!raw.ok()) return raw.status();
    const auto lease_id = first_lease_id + edge;
    auto lower = DeepSeekNcclBootstrapLease::Create(
        allocator, *raw, engine_epoch, edge, lease_id);
    if (!lower.ok()) {
      secure_zero(*raw);
      return lower.status();
    }
    auto upper = DeepSeekNcclBootstrapLease::Create(
        allocator, *raw, engine_epoch, edge, lease_id);
    secure_zero(*raw);
    if (!upper.ok()) return upper.status();
    if (lower->commitment() != upper->commitment()) {
      return Status::Internal(
          "DeepSeek NCCL endpoint bootstrap commitments differ");
    }
    const auto commitment = commitment_id(lower->commitment());
    if (commitment == 0) {
      return Status::Internal(
          "DeepSeek NCCL bootstrap commitment identity is zero");
    }
    issued.push_back(
        {edge, lease_id, commitment,
         std::make_unique<DeepSeekNcclBootstrapLease>(std::move(*lower)),
         std::make_unique<DeepSeekNcclBootstrapLease>(std::move(*upper))});
  }
  return issued;
}

}  // namespace pih
