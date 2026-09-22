#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "pih/model/runtime_profile_payload.h"

namespace pih {

enum class RuntimeProfileReferenceRole : std::uint8_t {
  kRuntimeSemantic = 1,
  kCapacityTemplate = 2,
  kFeatureSelection = 3,
  kReleaseEvidence = 4,
  kHardwareIdentity = 5,
  kKernelClosure = 6,
  kSecurityRuntime = 7,
};

struct RuntimeProfileReferenceDescriptor final {
  RuntimeProfileReferenceRole role =
      RuntimeProfileReferenceRole::kRuntimeSemantic;
  std::string schema_abi;
  std::uint64_t exact_bytes = 0;
  Sha256Digest object_root{};
};

class RuntimeProfileReferenceClosure final {
 public:
  static Result<RuntimeProfileReferenceClosure> Create(
      std::vector<RuntimeProfileReferenceDescriptor> descriptors);

  [[nodiscard]] std::span<const RuntimeProfileReferenceDescriptor>
  descriptors() const noexcept { return descriptors_; }
  [[nodiscard]] std::span<const std::byte> canonical_bytes() const noexcept {
    return canonical_bytes_;
  }
  [[nodiscard]] const Sha256Digest& closure_root() const noexcept {
    return closure_root_;
  }

 private:
  RuntimeProfileReferenceClosure(
      std::vector<RuntimeProfileReferenceDescriptor> descriptors,
      std::vector<std::byte> canonical_bytes,
      Sha256Digest closure_root) noexcept;
  std::vector<RuntimeProfileReferenceDescriptor> descriptors_;
  std::vector<std::byte> canonical_bytes_;
  Sha256Digest closure_root_{};
};

Result<RuntimeProfileReferenceClosure> parse_runtime_profile_reference_closure(
    std::span<const std::byte> bytes);

Status verify_runtime_profile_reference_closure(
    const VerifiedRuntimeProfile& profile,
    const SignedProfileEnvelope& envelope,
    const RuntimeProfileReferenceClosure& closure);

Status verify_runtime_profile_reference_objects(
    const RuntimeProfileReferenceClosure& closure,
    std::span<const std::vector<std::byte>> ordered_objects);

}  // namespace pih
