#include "pih/model/engine_inet_diag_transport_adapter.h"

namespace pih {
namespace {

constexpr std::size_t kMaximumInetDiagMessages = 1024U * 1024U + 1U;

}  // namespace

Result<EngineInetDiagTransportAdapter>
EngineInetDiagTransportAdapter::Create(
    std::uint64_t namespace_identity,
    std::uint64_t maximum_dump_duration_ms,
    EngineInetDiagTransport& transport) {
  if (namespace_identity == 0 || maximum_dump_duration_ms == 0) {
    return Status::InvalidArgument(
        "inet_diag transport adapter configuration is invalid");
  }
  return EngineInetDiagTransportAdapter(
      namespace_identity, maximum_dump_duration_ms, transport);
}

Result<EngineNetworkPortRawSnapshot>
EngineInetDiagTransportAdapter::capture(std::uint64_t namespace_identity) {
  if (namespace_identity != namespace_identity_) {
    return Status::FailedPrecondition(
        "inet_diag transport namespace identity drifted");
  }
  auto state = transport_->namespace_state(namespace_identity);
  if (!state.ok()) return state.status();
  if (*state == EngineInetDiagNamespaceHandleState::kDestroyed) {
    return EngineNetworkPortRawSnapshot{
        namespace_identity_, false, true, {}};
  }
  if (*state != EngineInetDiagNamespaceHandleState::kPresent) {
    return Status::FailedPrecondition(
        "inet_diag namespace handle state is invalid");
  }

  auto ticket = transport_->begin_dump(namespace_identity,
                                        maximum_dump_duration_ms_);
  if (!ticket.ok()) return ticket.status();
  if (ticket->sequence == 0 || ticket->absolute_deadline_ns == 0) {
    return Status::FailedPrecondition(
        "inet_diag dump ticket is invalid");
  }
  auto decoder = EngineInetDiagMultipartDecoder::Create(ticket->sequence);
  if (!decoder.ok()) return decoder.status();

  for (std::size_t count = 0; count < kMaximumInetDiagMessages; ++count) {
    auto message = transport_->receive(*ticket);
    if (!message.ok()) return message.status();
    if (!message->has_value()) {
      return Status::Unavailable(
          "inet_diag transport ended before DONE");
    }
    const auto kind = message->value().kind;
    auto status = decoder->accept(message->value());
    if (!status.ok()) return status;
    if (kind == EngineInetDiagMultipartMessageKind::kDone) {
      auto rows = decoder->finish();
      if (!rows.ok()) return rows.status();
      return EngineNetworkPortRawSnapshot{
          namespace_identity_, true, false, std::move(*rows)};
    }
  }
  return Status::ResourceExhausted(
      "inet_diag transport exceeded the message limit");
}

}  // namespace pih
