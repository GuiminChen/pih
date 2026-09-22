#pragma once

#include <cstdint>

#include "pih/backend/cuda/verified_kernel_launcher.h"

namespace pih {

enum class CudaCopyKind : std::uint8_t { kHostToDevice = 0, kDeviceToHost, kDeviceToDevice };
enum class CudaCopyMemoryType : std::uint8_t { kRegisteredPinnedHost = 0, kDevice };
enum class CudaCopyPurpose : std::uint8_t { kInput = 0, kWeight, kResult, kDiagnostic, kSameRankMove };

struct CudaCopyEndpoint final {
  std::uintptr_t allocation_base;
  std::uint64_t allocation_bytes;
  std::uint64_t offset;
  std::uint64_t owner_id;
  std::uint64_t generation;
  CudaCopyMemoryType memory_type;
  std::uint32_t rank;
  std::int32_t device_or_numa;
};

class TypedCopyDriver {
 public:
  virtual ~TypedCopyDriver() = default;
  [[nodiscard]] virtual std::uintptr_t context_identity() const noexcept = 0;
  virtual Status copy(CudaCopyKind kind, std::uintptr_t destination,
                      std::uintptr_t source, std::uint64_t bytes,
                      DriverStreamHandle stream) = 0;
};

class CudaTypedCopyPlan final {
 public:
  static Result<CudaTypedCopyPlan> Create(
      std::uint64_t plan_id, CudaCopyPurpose purpose, CudaCopyKind kind,
      CudaCopyEndpoint source, CudaCopyEndpoint destination,
      std::uint64_t bytes, std::uint64_t required_alignment,
      std::uintptr_t primary_context_identity, DriverStreamHandle stream,
      std::uint64_t completion_event_generation);

  Status submit(TypedCopyDriver& driver);

  [[nodiscard]] bool no_op() const noexcept { return bytes_ == 0; }
  [[nodiscard]] bool submitted() const noexcept { return submitted_; }
  [[nodiscard]] CudaCopyKind kind() const noexcept { return kind_; }
  [[nodiscard]] CudaCopyPurpose purpose() const noexcept { return purpose_; }
  [[nodiscard]] std::uint64_t bytes() const noexcept { return bytes_; }
  [[nodiscard]] std::uint64_t plan_id() const noexcept { return plan_id_; }
  [[nodiscard]] std::uint64_t source_owner_id() const noexcept {
    return source_owner_id_;
  }
  [[nodiscard]] std::uint64_t source_generation() const noexcept {
    return source_generation_;
  }
  [[nodiscard]] std::uint64_t destination_owner_id() const noexcept {
    return destination_owner_id_;
  }
  [[nodiscard]] std::uint64_t destination_generation() const noexcept {
    return destination_generation_;
  }
  [[nodiscard]] std::uint64_t completion_event_generation() const noexcept {
    return completion_event_generation_;
  }
  [[nodiscard]] std::uintptr_t context_identity() const noexcept {
    return context_identity_;
  }
  [[nodiscard]] DriverStreamHandle stream() const noexcept { return stream_; }
  [[nodiscard]] std::uintptr_t source_address() const noexcept {
    return source_address_;
  }
  [[nodiscard]] std::uintptr_t destination_address() const noexcept {
    return destination_address_;
  }

 private:
  CudaTypedCopyPlan(std::uint64_t plan_id, CudaCopyPurpose purpose,
                    CudaCopyKind kind, std::uintptr_t source_address,
                    std::uint64_t source_owner_id,
                    std::uint64_t source_generation,
                    std::uintptr_t destination_address,
                    std::uint64_t destination_owner_id,
                    std::uint64_t destination_generation, std::uint64_t bytes,
                    std::uintptr_t context_identity, DriverStreamHandle stream,
                    std::uint64_t completion_event_generation)
      : plan_id_(plan_id), purpose_(purpose), kind_(kind),
        source_address_(source_address),
        source_owner_id_(source_owner_id), source_generation_(source_generation),
        destination_address_(destination_address),
        destination_owner_id_(destination_owner_id),
        destination_generation_(destination_generation), bytes_(bytes),
        context_identity_(context_identity), stream_(stream),
        completion_event_generation_(completion_event_generation) {}

  std::uint64_t plan_id_;
  CudaCopyPurpose purpose_;
  CudaCopyKind kind_;
  std::uintptr_t source_address_;
  std::uint64_t source_owner_id_;
  std::uint64_t source_generation_;
  std::uintptr_t destination_address_;
  std::uint64_t destination_owner_id_;
  std::uint64_t destination_generation_;
  std::uint64_t bytes_;
  std::uintptr_t context_identity_;
  DriverStreamHandle stream_;
  std::uint64_t completion_event_generation_;
  bool submitted_ = false;
};

}  // namespace pih
