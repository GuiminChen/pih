#include "pih/model/qwen3_bf16_activation_route.h"

#include <initializer_list>

namespace pih {
namespace {

QwenBf16ActivationRoute route(
    std::initializer_list<QwenBf16ActivationSlot> reads,
    std::initializer_list<QwenBf16ActivationSlot> writes) {
  QwenBf16ActivationRoute result;
  for (const auto slot : reads) result.reads[result.read_count++] = slot;
  for (const auto slot : writes) result.writes[result.write_count++] = slot;
  return result;
}

Result<QwenBf16ActivationRoute> route_for(QwenBf16ExecutionOp operation) {
  using Slot = QwenBf16ActivationSlot;
  switch (operation) {
    case QwenBf16ExecutionOp::kEmbedding:
      return route({Slot::kTokenIds}, {Slot::kHidden});
    case QwenBf16ExecutionOp::kPrepareRopeAngles:
      return route({Slot::kPositions},
                   {Slot::kRopeCosine, Slot::kRopeSine});
    case QwenBf16ExecutionOp::kInputRmsNorm:
    case QwenBf16ExecutionOp::kPostAttentionRmsNorm:
    case QwenBf16ExecutionOp::kFinalRmsNorm:
      return route({Slot::kHidden}, {Slot::kNormalized});
    case QwenBf16ExecutionOp::kQueryLinear:
      return route({Slot::kNormalized}, {Slot::kQuery});
    case QwenBf16ExecutionOp::kKeyLinear:
      return route({Slot::kNormalized}, {Slot::kKey});
    case QwenBf16ExecutionOp::kValueLinear:
      return route({Slot::kNormalized}, {Slot::kValue});
    case QwenBf16ExecutionOp::kQueryRmsNorm:
      return route({Slot::kQuery}, {Slot::kQuery});
    case QwenBf16ExecutionOp::kKeyRmsNorm:
      return route({Slot::kKey}, {Slot::kKey});
    case QwenBf16ExecutionOp::kRope:
      return route({Slot::kQuery, Slot::kKey, Slot::kRopeCosine,
                    Slot::kRopeSine},
                   {Slot::kQuery, Slot::kKey});
    case QwenBf16ExecutionOp::kKvAppend:
      return route({Slot::kKey, Slot::kValue, Slot::kKvSlotStates,
                    Slot::kKvAppendHandles, Slot::kKvTokenOffsets},
                   {Slot::kKvBacking, Slot::kDeviceError});
    case QwenBf16ExecutionOp::kPagedGqa:
      return route({Slot::kQuery, Slot::kKvBacking, Slot::kKvSlotStates,
                    Slot::kKvVisibleHandles},
                   {Slot::kAttention, Slot::kDeviceError});
    case QwenBf16ExecutionOp::kAttentionOutputLinear:
      return route({Slot::kAttention}, {Slot::kNormalized});
    case QwenBf16ExecutionOp::kAttentionResidual:
    case QwenBf16ExecutionOp::kMlpResidual:
      return route({Slot::kHidden, Slot::kNormalized}, {Slot::kHidden});
    case QwenBf16ExecutionOp::kGateLinear:
      return route({Slot::kNormalized}, {Slot::kGate});
    case QwenBf16ExecutionOp::kUpLinear:
      return route({Slot::kNormalized}, {Slot::kUp});
    case QwenBf16ExecutionOp::kSiluMul:
      return route({Slot::kGate, Slot::kUp}, {Slot::kGate});
    case QwenBf16ExecutionOp::kDownLinear:
      return route({Slot::kGate}, {Slot::kNormalized});
    case QwenBf16ExecutionOp::kLmHead:
      return route({Slot::kNormalized}, {Slot::kLogits});
    case QwenBf16ExecutionOp::kGreedyArgmax:
      return route({Slot::kLogits},
                   {Slot::kSampledToken, Slot::kDeviceError});
  }
  return Status::InvalidArgument("unknown Qwen BF16 execution operation");
}

}  // namespace

Result<QwenBf16ActivationRoutePlan> QwenBf16ActivationRoutePlan::Create(
    const QwenBf16ExecutionSchedule& schedule) {
  QwenBf16ActivationRoutePlan plan;
  for (std::size_t step = 0; step < schedule.size(); ++step) {
    auto resolved = route_for(schedule[step].operation);
    if (!resolved.ok()) return resolved.status();
    plan.routes_[step] = *resolved;
  }
  return plan;
}

}  // namespace pih
