#pragma once

#include <memory>

#include "pih/model/runtime_profile_reference_closure.h"

namespace pih {

class RuntimeProfileReferenceLeaseOwner {
 public:
  virtual ~RuntimeProfileReferenceLeaseOwner() = default;
};

struct RuntimeProfileReferenceLeaseObservation final {
  RuntimeProfileReferenceRole role =
      RuntimeProfileReferenceRole::kRuntimeSemantic;
  std::uint64_t exact_bytes = 0;
  Sha256Digest content_root{};
  bool regular_file = false;
  bool immutable = false;
  std::shared_ptr<const RuntimeProfileReferenceLeaseOwner> owner;
};

class RuntimeProfileReferenceLeaseProbe {
 public:
  virtual ~RuntimeProfileReferenceLeaseProbe() = default;
  virtual Result<RuntimeProfileReferenceLeaseObservation> observe(
      const RuntimeProfileReferenceDescriptor& descriptor) = 0;
};

class VerifiedRuntimeProfileReferenceLeases final {
 public:
  [[nodiscard]] std::size_t size() const noexcept { return owners_.size(); }
  [[nodiscard]] const Sha256Digest& closure_root() const noexcept {
    return closure_root_;
  }
  [[nodiscard]] const RuntimeProfileRoots& reference_roots() const noexcept {
    return reference_roots_;
  }
  [[nodiscard]] std::uint64_t capacity_template_exact_bytes() const noexcept {
    return capacity_template_exact_bytes_;
  }

 private:
  friend Result<VerifiedRuntimeProfileReferenceLeases>
  verify_runtime_profile_reference_leases(
      const RuntimeProfileReferenceClosure&,
      RuntimeProfileReferenceLeaseProbe&);
  explicit VerifiedRuntimeProfileReferenceLeases(
      Sha256Digest closure_root, RuntimeProfileRoots reference_roots,
      std::uint64_t capacity_template_exact_bytes,
      std::vector<std::shared_ptr<const RuntimeProfileReferenceLeaseOwner>>
          owners) noexcept;
  Sha256Digest closure_root_{};
  RuntimeProfileRoots reference_roots_{};
  std::uint64_t capacity_template_exact_bytes_ = 0;
  std::vector<std::shared_ptr<const RuntimeProfileReferenceLeaseOwner>> owners_;
};

Result<VerifiedRuntimeProfileReferenceLeases>
verify_runtime_profile_reference_leases(
    const RuntimeProfileReferenceClosure& closure,
    RuntimeProfileReferenceLeaseProbe& probe);

}  // namespace pih
