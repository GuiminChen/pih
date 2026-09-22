#pragma once

#include "pih/model/engine_network_port_owner_classifier.h"

namespace pih {

enum class EngineInetDiagMultipartMessageKind : std::uint8_t {
  kRow,
  kDone,
  kError,
};

struct EngineInetDiagMultipartMessage final {
  std::uint32_t sequence = 0;
  EngineInetDiagMultipartMessageKind kind =
      EngineInetDiagMultipartMessageKind::kRow;
  bool multipart = false;
  bool truncated = false;
  std::int32_t error_code = 0;
  EngineNetworkPortCensusRow row{};
};

class EngineInetDiagMultipartDecoder final {
 public:
  static Result<EngineInetDiagMultipartDecoder> Create(
      std::uint32_t request_sequence);

  Status accept(const EngineInetDiagMultipartMessage& message);
  Result<std::vector<EngineNetworkPortCensusRow>> finish() const;

 private:
  explicit EngineInetDiagMultipartDecoder(std::uint32_t sequence) noexcept
      : sequence_(sequence) {}

  std::uint32_t sequence_ = 0;
  std::vector<EngineNetworkPortCensusRow> rows_;
  bool done_ = false;
  bool poisoned_ = false;
};

}  // namespace pih
