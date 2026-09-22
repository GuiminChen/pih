#include "expert_chain.h"
#include <vector>

namespace pih::deepseek_v41 {
namespace {
bool Same(EngramDeviceRegion a, EngramDeviceRegion b) {
  return a.address == b.address && a.bytes == b.bytes;
}
bool Overlap(EngramDeviceRegion a, EngramDeviceRegion b) {
  return a.bytes && b.bytes && a.address < b.address + b.bytes && b.address < a.address + a.bytes;
}
ExpertScatterLaunch Scatter(const ExpertTokenChainLaunch& x) {
  return {x.gather.dispatch, x.expert ? x.expert->down.output : EngramDeviceRegion{},
          x.accumulator, x.gather.expert, x.gather.rows};
}
}
Status ValidateExpertTokenChain(const ExpertTokenChainLaunch& x) {
  const auto gather = ValidateExpertGather(x.gather); if (!gather.ok()) return gather;
  if (x.expert.has_value() != (x.gather.rows != 0))
    return Status::InvalidArgument("Expert token chain requires execution exactly for nonempty experts");
  if (x.expert) {
    const auto expert = ValidateRoutedExpert(*x.expert); if (!expert.ok()) return expert;
    if (x.expert->gate.rows != x.gather.rows || x.expert->gate.stream != x.gather.dispatch.stream ||
        !Same(x.expert->gate.error_flag, x.gather.dispatch.error_flag) ||
        !Same(x.expert->gate.input, x.gather.output) ||
        !Same(x.expert->activation.route_weights, x.gather.gathered_weights))
      return Status::InvalidArgument("Expert token chain connection mismatch");
  }
  const auto scatter = ValidateExpertScatter(Scatter(x)); if (!scatter.ok()) return scatter;
  const auto& g = x.gather; const auto& d = g.dispatch;
  std::vector<EngramDeviceRegion> reads{d.indices, d.counts, d.slots, g.input, g.route_weights};
  std::vector<EngramDeviceRegion> writes{g.output, g.gathered_weights, x.accumulator, d.error_flag};
  if (x.expert) {
    for (const auto* projection : {&x.expert->gate, &x.expert->up, &x.expert->down}) {
      reads.push_back(projection->weight); reads.push_back(projection->weight_scales);
      writes.push_back(projection->quantized); writes.push_back(projection->activation_scales);
      writes.push_back(projection->output);
    }
    writes.push_back(x.expert->activation.output);
  }
  for (std::size_t i = 0; i < writes.size(); ++i) {
    for (const auto read : reads) if (Overlap(writes[i], read))
      return Status::InvalidArgument("Expert token chain overwrites live input or weight");
    for (std::size_t j = 0; j < i; ++j) if (Overlap(writes[i], writes[j]))
      return Status::InvalidArgument("Expert token chain writable alias");
  }
  return Status::Ok();
}
Status LaunchExpertTokenChain(const ExpertTokenChainLaunch& x) {
  const auto validation = ValidateExpertTokenChain(x); if (!validation.ok()) return validation;
  const auto gather = LaunchExpertGather(x.gather); if (!gather.ok()) return gather;
  if (x.expert) {
    const auto expert = LaunchRoutedExpert(*x.expert); if (!expert.ok()) return expert;
  }
  return LaunchExpertScatter(Scatter(x));
}
}  // namespace pih::deepseek_v41
