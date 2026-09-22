#include "inference_memory.h"
#include <limits>

namespace pih::deepseek_v41 {
namespace {
bool Valid(EngramDeviceRegion r, std::uint64_t bytes) {
  return r.address && r.address % 256 == 0 && r.bytes == bytes && bytes &&
      bytes <= std::numeric_limits<std::uintptr_t>::max() - r.address;
}
bool Overlap(EngramDeviceRegion a, EngramDeviceRegion b) {
  return a.bytes && b.bytes && a.address < b.address + b.bytes && b.address < a.address + a.bytes;
}
}
Result<InferenceMemoryPlan> InferenceMemoryPlan::Create(const FlashConfig& config, std::uint32_t tokens,
    std::uint32_t maximum, std::uint32_t world, std::uint32_t rank,
    std::uint64_t device_budget, std::uint64_t host_budget) {
  if (maximum < tokens || maximum > FlashConfig::kMaximumPositions)
    return Status::InvalidArgument("Inference memory context capacity invalid");
  InferenceMemoryPlan p;
  auto boundary = BoundaryPlan::Create(config, tokens, world, rank, device_budget, host_budget);
  if (!boundary.ok()) return boundary.status(); p.boundary_.emplace(std::move(*boundary));
  auto phases = StepPhasesPlan::Create(config, tokens, device_budget);
  if (!phases.ok()) return phases.status(); p.phases_.emplace(std::move(*phases));
  auto engram = EngramPlan::Create(config, tokens, world, rank, device_budget);
  if (!engram.ok()) return engram.status(); p.engram_.emplace(std::move(*engram));
  auto cache = SequenceCachePlan::Create(config, maximum, device_budget);
  if (!cache.ok()) return cache.status(); p.cache_.emplace(std::move(*cache));
  auto ffn = FfnPlan::Create(config, tokens, world, rank, device_budget);
  if (!ffn.ok()) return ffn.status(); p.ffn_.emplace(std::move(*ffn));
  auto attention = AttentionPreparePlan::Create(config, tokens, world, rank, device_budget);
  if (!attention.ok()) return attention.status(); p.attention_.emplace(std::move(*attention));
  auto output = AttentionOutputPlan::Create(config, tokens, maximum, world, rank, device_budget);
  if (!output.ok()) return output.status(); p.output_.emplace(std::move(*output));
  auto compressor = CompressorPlan::Create(config, tokens, device_budget);
  if (!compressor.ok()) return compressor.status(); p.compressor_.emplace(std::move(*compressor));
  auto indexer = IndexerPlan::Create(config, tokens, maximum, world, rank, device_budget);
  if (!indexer.ok()) return indexer.status(); p.indexer_.emplace(std::move(*indexer));
  const std::array sizes{p.boundary_->device_bytes(), p.phases_->bytes(), p.engram_->bytes(), p.cache_->bytes(),
      p.ffn_->bytes(), p.ffn_->bytes(), p.attention_->bytes(), p.output_->bytes(), p.compressor_->bytes(),
      p.indexer_->bytes(), p.indexer_->bytes(), p.indexer_->bytes(), p.indexer_->bytes(),
      p.indexer_->bytes(), p.indexer_->bytes(), p.indexer_->bytes(), p.indexer_->bytes()};
  for (std::size_t i = 0; i < sizes.size(); ++i) {
    if (sizes[i] > std::numeric_limits<std::uint64_t>::max() - 255)
      return Status::ResourceExhausted("Inference arena size overflow");
    const auto aligned = (sizes[i] + 255) & ~std::uint64_t{255};
    if (aligned > device_budget - p.bytes_) return Status::ResourceExhausted("Combined inference arenas exceed device budget");
    p.segments_[i] = {p.bytes_, sizes[i]}; p.bytes_ += aligned;
  }
  return p;
}
Result<InferenceMemoryViews> InferenceMemoryPlan::Bind(EngramDeviceRegion device, EngramDeviceRegion host,
    const BackboneWeightUpload& weights, ExpertWorkspaceOwner& workspace,
    std::uintptr_t communicator, std::uintptr_t event) const {
  if (!Valid(device, bytes_) || !Valid(host, host_bytes()) || Overlap(device, host) || !event ||
      weights.catalog().config_sha256() != boundary_->config_sha256() ||
      weights.catalog().world_size() != ffn_->world_size() || weights.catalog().rank() != ffn_->rank())
    return Status::InvalidArgument("Inference memory arena, event or uploaded identity mismatch");
  auto weight_arena = weights.Arena(); if (!weight_arena.ok()) return weight_arena.status();
  const auto& allocation = workspace.allocation_record();
  const EngramDeviceRegion expert{allocation.address, allocation.bytes};
  if (!expert.address || !expert.bytes || expert.bytes > std::numeric_limits<std::uintptr_t>::max() - expert.address ||
      Overlap(device, expert) || Overlap(host, expert) || Overlap(host, *weight_arena))
    return Status::InvalidArgument("Inference memory overlaps weights/workspace or workspace is absent");
  const auto safe = weights.ValidateScratch(std::array{device}); if (!safe.ok()) return safe;
  const auto r = [&](unsigned i) -> EngramDeviceRegion {
    return {device.address + segments_[i].offset, segments_[i].bytes};
  };
  // Completion/count addresses are filled from BoundaryPlan by InferenceOperation.
  BackboneResources resources{*phases_, *engram_, *cache_, *ffn_, *attention_, *output_, *compressor_, *indexer_,
      weights, workspace, r(1), r(2), r(3), {r(4), r(5)}, r(6), r(7), r(8),
      {r(9), r(10), r(11), r(12), r(13), r(14), r(15), r(16)}, {}, communicator, {{}, event}};
  return InferenceMemoryViews{resources, r(0), host};
}
}  // namespace pih::deepseek_v41
