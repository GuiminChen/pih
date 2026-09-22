#include "boundary_plan.h"
#include <limits>

namespace pih::deepseek_v41 {
namespace {
enum DeviceSlot : unsigned { kIds, kHash1, kHash14, kHidden, kResidual, kPre, kCollapse,
  kNorm, kLocalLogits, kLogits, kScores, kSortIds, kWeights, kReduction, kStats, kCandidate, kError };
enum HostSlot : unsigned { kHostIds, kHostHash1, kHostHash14, kHostError, kHostCounts, kHostCandidate };
constexpr std::array<std::uint64_t, 8> kStrides{4, 96, 96, 10240, 40960, 16, 10240, 10240};
bool Valid(EngramDeviceRegion r, std::uint64_t bytes) {
  return r.address && r.address % 256 == 0 && r.bytes == bytes &&
      r.bytes <= std::numeric_limits<std::uintptr_t>::max() - r.address;
}
}
Result<BoundaryPlan> BoundaryPlan::Create(const FlashConfig& config, std::uint32_t tokens,
    std::uint32_t world, std::uint32_t rank, std::uint64_t device_budget, std::uint64_t host_budget) {
  if (config.config_sha256() == Sha256Digest{} || !tokens || tokens > 4096 ||
      (world != 1 && world != 2 && world != 4 && world != 8) || rank >= world)
    return Status::InvalidArgument("Boundary plan configuration, token capacity or rank invalid");
  BoundaryPlan plan; plan.tokens_ = tokens; plan.world_ = world; plan.rank_ = rank;
  plan.config_sha256_ = config.config_sha256();
  const auto reserve = [](CacheSegment& slot, std::uint64_t bytes, std::uint64_t budget,
                          std::uint64_t& total) -> Status {
    const auto aligned = (bytes + 255) & ~std::uint64_t{255};
    if (aligned > budget - total) return Status::ResourceExhausted("Boundary arena exceeds memory budget");
    slot = {total, bytes}; total += aligned; return Status::Ok();
  };
  for (std::size_t i = 0; i < plan.device_.size(); ++i) {
    std::uint64_t bytes = 0;
    if (i < kStrides.size()) bytes = tokens * kStrides[i];
    else if (i == kLocalLogits) bytes = 129280ULL / world * 4;
    else if (i == kLogits) bytes = 129280ULL * 4;
    else if (i >= kScores && i <= kReduction) bytes = 131072ULL * 4;
    else if (i == kStats) bytes = 8;
    else if (i == kCandidate) bytes = sizeof(SamplingCandidate);
    else bytes = 4;
    auto status = reserve(plan.device_[i], bytes, device_budget, plan.device_bytes_); if (!status.ok()) return status;
  }
  const std::array host_sizes{tokens * 4ULL, tokens * 96ULL, tokens * 96ULL,
      4ULL, 384ULL / world * 4, static_cast<unsigned long long>(sizeof(SamplingCandidate))};
  for (std::size_t i = 0; i < plan.host_.size(); ++i) {
    auto status = reserve(plan.host_[i], host_sizes[i], host_budget, plan.host_bytes_); if (!status.ok()) return status;
  }
  return plan;
}
Result<BoundaryViews> BoundaryPlan::Bind(EngramDeviceRegion device, EngramDeviceRegion host,
    std::uint32_t tokens, std::uintptr_t stream) const {
  if (!stream || !tokens || tokens > tokens_ || !Valid(device, device_bytes_) || !Valid(host, host_bytes_) ||
      (device.address < host.address + host.bytes && host.address < device.address + device.bytes))
    return Status::InvalidArgument("Boundary arena extent/alignment, token prefix or stream invalid");
  const auto d = [&](unsigned i) -> EngramDeviceRegion {
    return {device.address + device_[i].offset, i < kStrides.size() ? tokens * kStrides[i] : device_[i].bytes};
  };
  const auto h = [&](unsigned i) -> EngramDeviceRegion {
    return {host.address + host_[i].offset, i < 3 ? tokens * (i == 0 ? 4ULL : 96ULL) : host_[i].bytes};
  };
  BoundaryViews x;
  const auto error = d(kError);
  x.embedding = {d(kIds), {}, d(kHidden), d(kResidual), d(kPre), error, stream, tokens, world_, rank_};
  x.input_upload = {d(kIds), h(kHostIds), {d(kHash1), d(kHash14)}, {h(kHostHash1), h(kHostHash14)}, stream};
  x.head.input.collapse = {d(kResidual), d(kPre), d(kCollapse), error, stream, tokens};
  x.head.input.norm = {d(kCollapse), {}, d(kNorm), error, EngramStorage::kBF16, stream, tokens, 5120};
  x.head.projection = {{d(kNorm).address + (tokens - 1ULL) * 10240, 10240}, {},
      d(kLocalLogits), error, stream, world_, rank_};
  x.logits = d(kLogits);
  x.sampling = {x.logits, d(kScores), d(kSortIds), d(kWeights), d(kReduction), d(kStats), d(kCandidate), error, stream, {}};
  x.host_error = h(kHostError); x.host_counts = h(kHostCounts); x.host_candidate = h(kHostCandidate);
  return x;
}
}  // namespace pih::deepseek_v41
