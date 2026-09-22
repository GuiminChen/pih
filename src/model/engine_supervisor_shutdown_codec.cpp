#include "pih/model/engine_supervisor_shutdown_codec.h"

#include <type_traits>

namespace pih {
namespace {

constexpr std::uint32_t kMagic = 0x44534958U;
constexpr std::uint16_t kRequestType = 1;
constexpr std::uint16_t kAckType = 2;
constexpr std::uint16_t kVersion = 1;
constexpr std::size_t kCodeOffset = 40;
constexpr std::size_t kReservedOffset = 41;

template <class T>
void put(std::array<std::byte, kEngineSupervisorShutdownFrameBytes>& output,
         std::size_t& offset, T value) {
  using U = std::make_unsigned_t<T>;
  const auto bits = static_cast<U>(value);
  for (std::size_t index = 0; index < sizeof(T); ++index) {
    output[offset++] = static_cast<std::byte>(
        (bits >> (index * 8U)) & static_cast<U>(0xffU));
  }
}

template <class T>
T get(std::span<const std::byte> input, std::size_t& offset) {
  using U = std::make_unsigned_t<T>;
  U value = 0;
  for (std::size_t index = 0; index < sizeof(T); ++index) {
    value |= static_cast<U>(std::to_integer<unsigned>(input[offset++]))
             << (index * 8U);
  }
  return static_cast<T>(value);
}

void encode_header(
    std::array<std::byte, kEngineSupervisorShutdownFrameBytes>& output,
    std::size_t& offset, std::uint16_t type) {
  put(output, offset, kMagic);
  put(output, offset, type);
  put(output, offset, kVersion);
}

Status validate_header(std::span<const std::byte> input,
                       std::uint16_t expected_type, std::size_t& offset) {
  if (input.size() != kEngineSupervisorShutdownFrameBytes) {
    return Status::InvalidArgument(
        "engine supervisor shutdown frame size is invalid");
  }
  if (get<std::uint32_t>(input, offset) != kMagic ||
      get<std::uint16_t>(input, offset) != expected_type ||
      get<std::uint16_t>(input, offset) != kVersion) {
    return Status::InvalidArgument(
        "engine supervisor shutdown frame header is invalid");
  }
  for (std::size_t index = kReservedOffset; index < input.size(); ++index) {
    if (input[index] != std::byte{0}) {
      return Status::InvalidArgument(
          "engine supervisor shutdown reserved bytes are nonzero");
    }
  }
  return Status::Ok();
}

bool valid_domain(std::uint64_t generation, std::uint64_t epoch,
                  std::uint64_t identity, std::uint64_t timestamp) {
  return generation != 0 && epoch != 0 && identity != 0 && timestamp != 0;
}

}  // namespace

std::array<std::byte, kEngineSupervisorShutdownFrameBytes>
encode_engine_supervisor_shutdown_request(
    const EngineSupervisorShutdownRequest& value) {
  std::array<std::byte, kEngineSupervisorShutdownFrameBytes> output{};
  std::size_t offset = 0;
  encode_header(output, offset, kRequestType);
  put(output, offset, value.engine_generation);
  put(output, offset, value.engine_epoch);
  put(output, offset, value.request_identity);
  put(output, offset, value.deadline_ns);
  output[kCodeOffset] = static_cast<std::byte>(value.kind);
  return output;
}

Result<EngineSupervisorShutdownRequest>
decode_engine_supervisor_shutdown_request(std::span<const std::byte> input) {
  std::size_t offset = 0;
  const auto header = validate_header(input, kRequestType, offset);
  if (!header.ok()) return header;
  EngineSupervisorShutdownRequest value;
  value.engine_generation = get<std::uint64_t>(input, offset);
  value.engine_epoch = get<std::uint64_t>(input, offset);
  value.request_identity = get<std::uint64_t>(input, offset);
  value.deadline_ns = get<std::uint64_t>(input, offset);
  value.kind = static_cast<EngineSupervisorShutdownRequestKind>(
      std::to_integer<std::uint8_t>(input[kCodeOffset]));
  if (!valid_domain(value.engine_generation, value.engine_epoch,
                    value.request_identity, value.deadline_ns) ||
      value.kind != EngineSupervisorShutdownRequestKind::kForceStop) {
    return Status::InvalidArgument(
        "engine supervisor shutdown request is invalid");
  }
  return value;
}

std::array<std::byte, kEngineSupervisorShutdownFrameBytes>
encode_engine_supervisor_shutdown_ack(
    const EngineSupervisorShutdownAck& value) {
  std::array<std::byte, kEngineSupervisorShutdownFrameBytes> output{};
  std::size_t offset = 0;
  encode_header(output, offset, kAckType);
  put(output, offset, value.engine_generation);
  put(output, offset, value.engine_epoch);
  put(output, offset, value.request_identity);
  put(output, offset, value.acknowledged_ns);
  output[kCodeOffset] = static_cast<std::byte>(value.disposition);
  return output;
}

Result<EngineSupervisorShutdownAck> decode_engine_supervisor_shutdown_ack(
    std::span<const std::byte> input) {
  std::size_t offset = 0;
  const auto header = validate_header(input, kAckType, offset);
  if (!header.ok()) return header;
  EngineSupervisorShutdownAck value;
  value.engine_generation = get<std::uint64_t>(input, offset);
  value.engine_epoch = get<std::uint64_t>(input, offset);
  value.request_identity = get<std::uint64_t>(input, offset);
  value.acknowledged_ns = get<std::uint64_t>(input, offset);
  value.disposition = static_cast<EngineSupervisorShutdownAckDisposition>(
      std::to_integer<std::uint8_t>(input[kCodeOffset]));
  const bool valid_disposition =
      value.disposition == EngineSupervisorShutdownAckDisposition::kAccepted ||
      value.disposition ==
          EngineSupervisorShutdownAckDisposition::kAlreadyStopping ||
      value.disposition == EngineSupervisorShutdownAckDisposition::kRejected;
  if (!valid_domain(value.engine_generation, value.engine_epoch,
                    value.request_identity, value.acknowledged_ns) ||
      !valid_disposition) {
    return Status::InvalidArgument(
        "engine supervisor shutdown acknowledgement is invalid");
  }
  return value;
}

Result<EngineSupervisorShutdownRequestGate>
EngineSupervisorShutdownRequestGate::Create(std::uint64_t engine_generation,
                                            std::uint64_t engine_epoch) {
  if (engine_generation == 0 || engine_epoch == 0) {
    return Status::InvalidArgument(
        "engine supervisor shutdown request gate identity is invalid");
  }
  return EngineSupervisorShutdownRequestGate(engine_generation, engine_epoch);
}

Result<EngineSupervisorShutdownAckDisposition>
EngineSupervisorShutdownRequestGate::accept(
    const EngineSupervisorShutdownRequest& request, std::uint64_t now_ns) {
  if (poisoned_) {
    return Status::FailedPrecondition(
        "engine supervisor shutdown request gate is poisoned");
  }
  const bool domain_matches =
      request.engine_generation == engine_generation_ &&
      request.engine_epoch == engine_epoch_ && request.request_identity == 1 &&
      request.deadline_ns != 0 &&
      request.kind == EngineSupervisorShutdownRequestKind::kForceStop;
  if (!accepted_) {
    if (!domain_matches || now_ns >= request.deadline_ns) {
      poisoned_ = true;
      return Status::FailedPrecondition(
          "engine supervisor shutdown request identity or deadline is invalid");
    }
    accepted_request_ = request;
    accepted_ = true;
    return EngineSupervisorShutdownAckDisposition::kAccepted;
  }
  const bool exact_replay =
      domain_matches &&
      request.request_identity == accepted_request_.request_identity &&
      request.deadline_ns == accepted_request_.deadline_ns &&
      request.kind == accepted_request_.kind;
  if (!exact_replay) {
    poisoned_ = true;
    return Status::FailedPrecondition(
        "engine supervisor shutdown request replay drifted");
  }
  return EngineSupervisorShutdownAckDisposition::kAlreadyStopping;
}

Result<EngineSupervisorShutdownAckGate>
EngineSupervisorShutdownAckGate::Create(std::uint64_t engine_generation,
                                        std::uint64_t engine_epoch,
                                        std::uint64_t request_identity) {
  if (engine_generation == 0 || engine_epoch == 0 || request_identity == 0) {
    return Status::InvalidArgument(
        "engine supervisor shutdown acknowledgement identity is invalid");
  }
  return EngineSupervisorShutdownAckGate(engine_generation, engine_epoch,
                                         request_identity);
}

Status EngineSupervisorShutdownAckGate::accept(
    const EngineSupervisorShutdownAck& ack) {
  if (poisoned_ || accepted_) {
    return Status::FailedPrecondition(
        "engine supervisor shutdown acknowledgement gate is terminal");
  }
  if (ack.engine_generation != engine_generation_ ||
      ack.engine_epoch != engine_epoch_ ||
      ack.request_identity != request_identity_ || ack.acknowledged_ns == 0) {
    poisoned_ = true;
    return Status::FailedPrecondition(
        "engine supervisor shutdown acknowledgement identity drifted");
  }
  if (ack.disposition == EngineSupervisorShutdownAckDisposition::kRejected) {
    poisoned_ = true;
    return Status::FailedPrecondition(
        "engine supervisor rejected shutdown escalation");
  }
  if (ack.disposition != EngineSupervisorShutdownAckDisposition::kAccepted &&
      ack.disposition !=
          EngineSupervisorShutdownAckDisposition::kAlreadyStopping) {
    poisoned_ = true;
    return Status::FailedPrecondition(
        "engine supervisor shutdown acknowledgement disposition is invalid");
  }
  accepted_ = true;
  return Status::Ok();
}

}  // namespace pih
