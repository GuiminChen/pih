#include "ffn_route.h"

namespace pih::deepseek_v41 {
namespace {
bool Same(EngramDeviceRegion a, EngramDeviceRegion b) { return a.address == b.address && a.bytes == b.bytes; }
bool Overlap(EngramDeviceRegion a, EngramDeviceRegion b) {
  return a.bytes && b.bytes && a.address < b.address + b.bytes && b.address < a.address + a.bytes;
}
}
Status ValidateFfnRoute(const FlashConfig& config, const FfnRouteLaunch& x) {
  if (config.config_sha256() == Sha256Digest{} || x.router.layer >= FlashConfig::kMainLayers)
    return Status::InvalidArgument("FFN route requires admitted backbone configuration and layer");
  const auto input = ValidateMhcSublayerInput(x.input); if (!input.ok()) return input;
  const auto router = ValidateRouter(x.router); if (!router.ok()) return router;
  const auto dispatch = ValidateExpertDispatch(x.dispatch); if (!dispatch.ok()) return dispatch;
  const auto& m = x.input.mix; const auto& c = x.input.input.collapse; const auto& n = x.input.input.norm;
  const auto& r = x.router; const auto& d = x.dispatch;
  const auto& layer = config.attention_sharing()[r.layer];
  if (layer.routed_experts != 384 || layer.activated_experts != 6 ||
      m.tokens != r.tokens || r.tokens != d.tokens || r.layer != d.layer ||
      m.stream != r.stream || r.stream != d.stream || !Same(n.output, r.input) ||
      !Same(r.indices, d.indices) || !Same(m.error_flag, r.error_flag) || !Same(r.error_flag, d.error_flag))
    return Status::InvalidArgument("FFN route layer, step or buffer connection mismatch");
  const std::array reads{m.residual, m.fn, m.scale, m.base, c.pre, n.weight,
      r.weight, r.bias, r.image_bias, r.image_mask};
  const std::array writes{m.pre, m.post, m.comb, c.output, n.output,
      r.logits, r.indices, r.route_weights, d.counts, d.slots, d.error_flag};
  for (std::size_t i = 0; i < writes.size(); ++i) {
    for (const auto read : reads) if (Overlap(writes[i], read))
      return Status::InvalidArgument("FFN routing overwrites live input or weight");
    for (std::size_t j = 0; j < i; ++j) if (Overlap(writes[i], writes[j]))
      return Status::InvalidArgument("FFN routing writable alias");
  }
  return Status::Ok();
}
Result<ExpertCounts> LaunchFfnRoute(const FlashConfig& config, const FfnRouteLaunch& x,
    EngramDeviceRegion host_counts, const EngramCompletionResources& resources,
    ExpertCounts::Clock::time_point deadline) {
  const auto validation = ValidateFfnRoute(config, x); if (!validation.ok()) return validation;
  if (ExpertCounts::Clock::now() >= deadline) return Status::DeadlineExceeded("FFN route deadline expired");
  const auto resources_ready = ValidateEngramCompletionResources(resources); if (!resources_ready.ok()) return resources_ready;
  const auto input = LaunchMhcSublayerInput(x.input); if (!input.ok()) return input;
  const auto router = LaunchRouter(x.router); if (!router.ok()) return router;
  return ExpertCounts::Start(x.dispatch, host_counts, resources, deadline);
}
}  // namespace pih::deepseek_v41
