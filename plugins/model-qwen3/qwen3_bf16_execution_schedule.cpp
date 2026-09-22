#include "pih/model/qwen3_bf16_execution_schedule.h"

#include <array>

namespace pih {
namespace {

bool is_official_config(const Qwen3Config& config) noexcept {
  return config.hidden_size == 1024 && config.intermediate_size == 3072 &&
         config.layers == 28 && config.attention_heads == 16 &&
         config.kv_heads == 8 && config.head_dim == 128 &&
         config.vocabulary_size == 151936 &&
         config.maximum_positions == 40960 &&
         config.rope_theta == 1'000'000.0 &&
         config.rms_norm_epsilon == 0.000001 &&
         config.bos_token_id == 151643 && config.eos_token_id == 151645;
}

constexpr std::array<QwenBf16ExecutionOp,
                     QwenBf16ExecutionSchedule::kOperationsPerLayer>
    kLayerOperations{
        QwenBf16ExecutionOp::kInputRmsNorm,
        QwenBf16ExecutionOp::kQueryLinear,
        QwenBf16ExecutionOp::kKeyLinear,
        QwenBf16ExecutionOp::kValueLinear,
        QwenBf16ExecutionOp::kQueryRmsNorm,
        QwenBf16ExecutionOp::kKeyRmsNorm,
        QwenBf16ExecutionOp::kRope,
        QwenBf16ExecutionOp::kKvAppend,
        QwenBf16ExecutionOp::kPagedGqa,
        QwenBf16ExecutionOp::kAttentionOutputLinear,
        QwenBf16ExecutionOp::kAttentionResidual,
        QwenBf16ExecutionOp::kPostAttentionRmsNorm,
        QwenBf16ExecutionOp::kGateLinear,
        QwenBf16ExecutionOp::kUpLinear,
        QwenBf16ExecutionOp::kSiluMul,
        QwenBf16ExecutionOp::kDownLinear,
        QwenBf16ExecutionOp::kMlpResidual,
    };

}  // namespace

Result<QwenBf16ExecutionSchedule> QwenBf16ExecutionSchedule::Create(
    const Qwen3Config& config) {
  if (!is_official_config(config)) {
    return Status::InvalidArgument(
        "Qwen BF16 execution schedule requires the frozen official "
        "Qwen3-0.6B architecture");
  }

  std::array<QwenBf16ExecutionStep, kStepCount> steps{};
  std::size_t next = 0;
  steps[next++] = {QwenBf16ExecutionOp::kEmbedding, kGlobalLayer};
  steps[next++] = {QwenBf16ExecutionOp::kPrepareRopeAngles, kGlobalLayer};
  for (std::uint32_t layer = 0; layer < kLayerCount; ++layer) {
    for (const auto operation : kLayerOperations) {
      steps[next++] = {operation, layer};
    }
  }
  steps[next++] = {QwenBf16ExecutionOp::kFinalRmsNorm, kGlobalLayer};
  steps[next++] = {QwenBf16ExecutionOp::kLmHead, kGlobalLayer};
  steps[next] = {QwenBf16ExecutionOp::kGreedyArgmax, kGlobalLayer};
  return QwenBf16ExecutionSchedule(steps);
}

Result<QwenBf16LinearKind> qwen_bf16_linear_kind(
    QwenBf16ExecutionOp operation) {
  switch (operation) {
    case QwenBf16ExecutionOp::kQueryLinear:
      return QwenBf16LinearKind::kQuery;
    case QwenBf16ExecutionOp::kKeyLinear:
      return QwenBf16LinearKind::kKey;
    case QwenBf16ExecutionOp::kValueLinear:
      return QwenBf16LinearKind::kValue;
    case QwenBf16ExecutionOp::kAttentionOutputLinear:
      return QwenBf16LinearKind::kAttentionOutput;
    case QwenBf16ExecutionOp::kGateLinear:
      return QwenBf16LinearKind::kGate;
    case QwenBf16ExecutionOp::kUpLinear:
      return QwenBf16LinearKind::kUp;
    case QwenBf16ExecutionOp::kDownLinear:
      return QwenBf16LinearKind::kDown;
    case QwenBf16ExecutionOp::kLmHead:
      return QwenBf16LinearKind::kLmHead;
    default:
      return Status::InvalidArgument(
          "Qwen BF16 execution operation is not a linear operation");
  }
}

}  // namespace pih
