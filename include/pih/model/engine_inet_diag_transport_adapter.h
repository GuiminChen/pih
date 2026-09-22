#pragma once

#include <optional>

#include "pih/model/engine_inet_diag_multipart_decoder.h"

namespace pih {

enum class EngineInetDiagNamespaceHandleState : std::uint8_t {
  kPresent,
  kDestroyed,
};

struct EngineInetDiagDumpTicket final {
  std::uint32_t sequence = 0;
  std::uint64_t absolute_deadline_ns = 0;
};

class EngineInetDiagTransport {
 public:
  virtual ~EngineInetDiagTransport() = default;
  virtual Result<EngineInetDiagNamespaceHandleState> namespace_state(
      std::uint64_t namespace_identity) = 0;
  virtual Result<EngineInetDiagDumpTicket> begin_dump(
      std::uint64_t namespace_identity,
      std::uint64_t maximum_duration_ms) = 0;
  virtual Result<std::optional<EngineInetDiagMultipartMessage>> receive(
      const EngineInetDiagDumpTicket& ticket) = 0;
};

class EngineInetDiagTransportAdapter final
    : public EngineNetworkPortCensusBackend {
 public:
  static Result<EngineInetDiagTransportAdapter> Create(
      std::uint64_t namespace_identity,
      std::uint64_t maximum_dump_duration_ms,
      EngineInetDiagTransport& transport);

  Result<EngineNetworkPortRawSnapshot> capture(
      std::uint64_t namespace_identity) override;

 private:
  EngineInetDiagTransportAdapter(
      std::uint64_t namespace_identity,
      std::uint64_t maximum_dump_duration_ms,
      EngineInetDiagTransport& transport) noexcept
      : namespace_identity_(namespace_identity),
        maximum_dump_duration_ms_(maximum_dump_duration_ms),
        transport_(&transport) {}

  std::uint64_t namespace_identity_ = 0;
  std::uint64_t maximum_dump_duration_ms_ = 0;
  EngineInetDiagTransport* transport_ = nullptr;
};

}  // namespace pih
