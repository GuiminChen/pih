#include "pih/model/qwen3_bf16_command_buffer.h"

#include <algorithm>
#include <array>
#include <span>

namespace pih {
namespace {

void write_u16(std::span<std::byte> wire, std::size_t offset,
               std::uint16_t value) {
  wire[offset] = static_cast<std::byte>(value);
  wire[offset + 1] = static_cast<std::byte>(value >> 8U);
}

void write_u32(std::span<std::byte> wire, std::size_t offset,
               std::uint32_t value) {
  for (std::size_t index = 0; index < 4; ++index) {
    wire[offset + index] = static_cast<std::byte>(value >> (index * 8U));
  }
}

Result<QwenBf16Primitive> kernel_primitive(QwenBf16ExecutionOp operation) {
  switch (operation) {
    case QwenBf16ExecutionOp::kEmbedding:
      return QwenBf16Primitive::kEmbedding;
    case QwenBf16ExecutionOp::kPrepareRopeAngles:
      return QwenBf16Primitive::kRopeAngles;
    case QwenBf16ExecutionOp::kInputRmsNorm:
    case QwenBf16ExecutionOp::kQueryRmsNorm:
    case QwenBf16ExecutionOp::kKeyRmsNorm:
    case QwenBf16ExecutionOp::kPostAttentionRmsNorm:
    case QwenBf16ExecutionOp::kFinalRmsNorm:
      return QwenBf16Primitive::kRmsNorm;
    case QwenBf16ExecutionOp::kRope:
      return QwenBf16Primitive::kRope;
    case QwenBf16ExecutionOp::kKvAppend:
      return QwenBf16Primitive::kKvAppend;
    case QwenBf16ExecutionOp::kPagedGqa:
      return QwenBf16Primitive::kPagedGqa;
    case QwenBf16ExecutionOp::kAttentionResidual:
    case QwenBf16ExecutionOp::kMlpResidual:
      return QwenBf16Primitive::kResidualAdd;
    case QwenBf16ExecutionOp::kSiluMul:
      return QwenBf16Primitive::kSiluMul;
    case QwenBf16ExecutionOp::kGreedyArgmax:
      return QwenBf16Primitive::kGreedyArgmax;
    default:
      return Status::InvalidArgument("Qwen execution operation is not a kernel");
  }
}

}  // namespace

QwenBf16CommandBuffer::CanonicalWire
QwenBf16CommandBuffer::canonical_wire() const noexcept {
  CanonicalWire wire{};
  constexpr std::array<std::byte, 8> kMagic{
      std::byte{'X'}, std::byte{'Q'}, std::byte{'C'}, std::byte{'B'},
      std::byte{1}, std::byte{0}, std::byte{0}, std::byte{0}};
  std::copy(kMagic.begin(), kMagic.end(), wire.begin());
  write_u32(wire, 8, static_cast<std::uint32_t>(kCommandCount));
  write_u32(wire, 12, static_cast<std::uint32_t>(kCanonicalCommandBytes));
  for (std::size_t index = 0; index < commands_.size(); ++index) {
    const auto& command = commands_[index];
    const std::size_t offset =
        kCanonicalHeaderBytes + index * kCanonicalCommandBytes;
    wire[offset] = static_cast<std::byte>(command.backend);
    wire[offset + 1] =
        static_cast<std::byte>(command.execution_step.operation);
    wire[offset + 2] = static_cast<std::byte>(command.primitive);
    wire[offset + 3] = static_cast<std::byte>(command.linear_kind);
    wire[offset + 4] = static_cast<std::byte>(command.subcommand);
    write_u16(wire, offset + 8, command.logical_step);
    write_u16(wire, offset + 10, command.tensor_index);
    write_u32(wire, offset + 12, command.execution_step.layer);
  }
  return wire;
}

Result<QwenBf16CommandBuffer> QwenBf16CommandBuffer::Create(
    const QwenBf16ExecutionSchedule& schedule,
    const QwenBf16WeightBindingPlan& weights) {
  QwenBf16CommandBuffer buffer;
  std::size_t next = 0;
  std::size_t kernels = 0;
  std::size_t linears = 0;
  for (std::size_t logical_step = 0; logical_step < schedule.size();
       ++logical_step) {
    buffer.offsets_[logical_step] = static_cast<std::uint16_t>(next);
    const auto step = schedule[logical_step];
    auto linear = qwen_bf16_linear_kind(step.operation);
    const auto* tensor = weights.tensor(logical_step);
    const auto tensor_index =
        tensor == nullptr ? QwenBf16PreparedCommand::kNoTensor
                          : weights[logical_step].tensor_index;
    if (linear.ok()) {
      buffer.commands_[next++] = {
          QwenBf16CommandBackend::kLinear,
          static_cast<std::uint16_t>(logical_step), 0, tensor_index,
          QwenBf16Primitive::kEmbedding, *linear, step};
      ++linears;
      continue;
    }
    auto primitive = kernel_primitive(step.operation);
    if (!primitive.ok()) return primitive.status();
    const std::uint8_t submissions =
        step.operation == QwenBf16ExecutionOp::kRope ? 2 : 1;
    for (std::uint8_t subcommand = 0; subcommand < submissions; ++subcommand) {
      buffer.commands_[next++] = {
          QwenBf16CommandBackend::kKernel,
          static_cast<std::uint16_t>(logical_step), subcommand, tensor_index,
          *primitive, QwenBf16LinearKind::kQuery, step};
      ++kernels;
    }
  }
  buffer.offsets_[schedule.size()] = static_cast<std::uint16_t>(next);
  if (next != kCommandCount || kernels != kKernelCommandCount ||
      linears != kLinearCommandCount) {
    return Status::FailedPrecondition(
        "Qwen execution command buffer does not match frozen cardinalities");
  }
  return buffer;
}

}  // namespace pih
