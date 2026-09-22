#include "pih/model/runtime_profile_reference_closure.h"

#include <algorithm>
#include <array>

namespace pih {
namespace {

constexpr std::array<std::byte, 8> kMagic{
    std::byte{'X'}, std::byte{'I'}, std::byte{'R'}, std::byte{'R'},
    std::byte{'E'}, std::byte{'F'}, std::byte{'1'}, std::byte{0}};
constexpr std::size_t kRoleCount = 7;
constexpr std::size_t kMaximumSchemaBytes = 96;

bool nonzero(const Sha256Digest& value) {
  return std::ranges::any_of(value.bytes,
      [](std::byte value) { return value != std::byte{0}; });
}

bool safe_schema(std::string_view value) {
  return !value.empty() && value.size() <= kMaximumSchemaBytes &&
         std::ranges::all_of(value, [](unsigned char character) {
           return (character >= 'a' && character <= 'z') ||
                  (character >= '0' && character <= '9') ||
                  character == '_';
         });
}

void append_u64(std::vector<std::byte>& output, std::uint64_t value) {
  for (unsigned shift = 0; shift < 64; shift += 8) {
    output.push_back(static_cast<std::byte>((value >> shift) & 0xffU));
  }
}

Result<std::uint64_t> read_u64(std::span<const std::byte> bytes,
                               std::size_t& cursor) {
  if (bytes.size() - cursor < 8) {
    return Status::InvalidArgument("runtime reference closure is truncated");
  }
  std::uint64_t value = 0;
  for (unsigned shift = 0; shift < 64; shift += 8) {
    value |= static_cast<std::uint64_t>(bytes[cursor++]) << shift;
  }
  return value;
}

const Sha256Digest& payload_root(const RuntimeProfileRoots& roots,
                                 RuntimeProfileReferenceRole role) {
  switch (role) {
    case RuntimeProfileReferenceRole::kRuntimeSemantic:
      return roots.runtime_semantic_root;
    case RuntimeProfileReferenceRole::kCapacityTemplate:
      return roots.capacity_template_root;
    case RuntimeProfileReferenceRole::kFeatureSelection:
      return roots.feature_selection_root;
    case RuntimeProfileReferenceRole::kReleaseEvidence:
      return roots.release_evidence_root;
    case RuntimeProfileReferenceRole::kHardwareIdentity:
      return roots.hardware_identity_root;
    case RuntimeProfileReferenceRole::kKernelClosure:
      return roots.kernel_closure_root;
    case RuntimeProfileReferenceRole::kSecurityRuntime:
      return roots.security_runtime_root;
  }
  return roots.runtime_semantic_root;
}

}  // namespace

RuntimeProfileReferenceClosure::RuntimeProfileReferenceClosure(
    std::vector<RuntimeProfileReferenceDescriptor> descriptors,
    std::vector<std::byte> canonical_bytes,
    Sha256Digest closure_root) noexcept
    : descriptors_(std::move(descriptors)),
      canonical_bytes_(std::move(canonical_bytes)),
      closure_root_(closure_root) {}

Result<RuntimeProfileReferenceClosure> RuntimeProfileReferenceClosure::Create(
    std::vector<RuntimeProfileReferenceDescriptor> descriptors) {
  if (descriptors.size() != kRoleCount) {
    return Status::InvalidArgument(
        "runtime reference closure must contain seven roles");
  }
  std::ranges::sort(descriptors, {},
                    &RuntimeProfileReferenceDescriptor::role);
  for (std::size_t left = 0; left < descriptors.size(); ++left) {
    for (std::size_t right = left + 1; right < descriptors.size(); ++right) {
      if (descriptors[left].object_root == descriptors[right].object_root) {
        return Status::InvalidArgument(
            "runtime reference object is reused across incompatible roles");
      }
    }
  }
  std::vector<std::byte> canonical(kMagic.begin(), kMagic.end());
  canonical.push_back(std::byte{1});
  canonical.push_back(static_cast<std::byte>(kRoleCount));
  for (std::size_t index = 0; index < descriptors.size(); ++index) {
    const auto& descriptor = descriptors[index];
    if (static_cast<std::uint8_t>(descriptor.role) != index + 1 ||
        !safe_schema(descriptor.schema_abi) || descriptor.exact_bytes == 0 ||
        !nonzero(descriptor.object_root)) {
      return Status::InvalidArgument(
          "runtime reference descriptor is invalid or duplicated");
    }
    canonical.push_back(static_cast<std::byte>(descriptor.role));
    canonical.push_back(static_cast<std::byte>(descriptor.schema_abi.size()));
    canonical.insert(canonical.end(),
                     reinterpret_cast<const std::byte*>(descriptor.schema_abi.data()),
                     reinterpret_cast<const std::byte*>(descriptor.schema_abi.data() +
                                                        descriptor.schema_abi.size()));
    append_u64(canonical, descriptor.exact_bytes);
    canonical.insert(canonical.end(), descriptor.object_root.bytes.begin(),
                     descriptor.object_root.bytes.end());
  }
  auto root = sha256(canonical);
  if (!root.ok()) return root.status();
  return RuntimeProfileReferenceClosure(std::move(descriptors),
                                        std::move(canonical), *root);
}

Result<RuntimeProfileReferenceClosure> parse_runtime_profile_reference_closure(
    std::span<const std::byte> bytes) {
  if (bytes.size() < kMagic.size() + 2 ||
      !std::ranges::equal(bytes.first(kMagic.size()), kMagic) ||
      bytes[kMagic.size()] != std::byte{1} ||
      bytes[kMagic.size() + 1] != static_cast<std::byte>(kRoleCount)) {
    return Status::InvalidArgument("runtime reference closure header is invalid");
  }
  std::size_t cursor = kMagic.size() + 2;
  std::vector<RuntimeProfileReferenceDescriptor> descriptors;
  descriptors.reserve(kRoleCount);
  for (std::size_t index = 0; index < kRoleCount; ++index) {
    if (bytes.size() - cursor < 2) {
      return Status::InvalidArgument("runtime reference closure is truncated");
    }
    RuntimeProfileReferenceDescriptor descriptor;
    descriptor.role = static_cast<RuntimeProfileReferenceRole>(bytes[cursor++]);
    const auto schema_bytes = static_cast<std::uint8_t>(bytes[cursor++]);
    if (schema_bytes == 0 || schema_bytes > kMaximumSchemaBytes ||
        bytes.size() - cursor < schema_bytes) {
      return Status::InvalidArgument("runtime reference schema is invalid");
    }
    descriptor.schema_abi.assign(
        reinterpret_cast<const char*>(bytes.data() + cursor), schema_bytes);
    cursor += schema_bytes;
    auto exact_bytes = read_u64(bytes, cursor);
    if (!exact_bytes.ok()) return exact_bytes.status();
    descriptor.exact_bytes = *exact_bytes;
    if (bytes.size() - cursor < descriptor.object_root.bytes.size()) {
      return Status::InvalidArgument("runtime reference root is truncated");
    }
    std::ranges::copy(bytes.subspan(cursor, descriptor.object_root.bytes.size()),
                      descriptor.object_root.bytes.begin());
    cursor += descriptor.object_root.bytes.size();
    descriptors.push_back(std::move(descriptor));
  }
  if (cursor != bytes.size()) {
    return Status::InvalidArgument("runtime reference closure has trailing bytes");
  }
  auto result = RuntimeProfileReferenceClosure::Create(std::move(descriptors));
  if (!result.ok() || !std::ranges::equal(result->canonical_bytes(), bytes)) {
    return Status::InvalidArgument("runtime reference closure is noncanonical");
  }
  return result;
}

Status verify_runtime_profile_reference_closure(
    const VerifiedRuntimeProfile& profile,
    const SignedProfileEnvelope& envelope,
    const RuntimeProfileReferenceClosure& closure) {
  if (!(closure.closure_root() == envelope.reference_closure_root())) {
    return Status::FailedPrecondition(
        "runtime reference closure differs from signed envelope");
  }
  for (const auto& descriptor : closure.descriptors()) {
    if (!(descriptor.object_root ==
          payload_root(profile.payload().roots(), descriptor.role))) {
      return Status::FailedPrecondition(
          "runtime reference descriptor differs from profile payload");
    }
  }
  return Status::Ok();
}

Status verify_runtime_profile_reference_objects(
    const RuntimeProfileReferenceClosure& closure,
    std::span<const std::vector<std::byte>> ordered_objects) {
  if (ordered_objects.size() != closure.descriptors().size()) {
    return Status::InvalidArgument(
        "runtime reference object count differs from closure");
  }
  for (std::size_t index = 0; index < ordered_objects.size(); ++index) {
    const auto& descriptor = closure.descriptors()[index];
    const auto& object = ordered_objects[index];
    if (object.size() != descriptor.exact_bytes) {
      return Status::FailedPrecondition(
          "runtime reference object length differs from descriptor");
    }
    auto root = sha256(object);
    if (!root.ok()) return root.status();
    if (!(*root == descriptor.object_root)) {
      return Status::FailedPrecondition(
          "runtime reference object digest differs from descriptor");
    }
  }
  return Status::Ok();
}

Result<RuntimeEngineAdmission> retain_runtime_engine_reference_objects(
    RuntimeEngineAdmission admission,
    const RuntimeProfileReferenceClosure& closure,
    std::vector<std::vector<std::byte>> objects) {
  if (!(closure.closure_root() == admission.reference_closure_root_)) {
    return Status::FailedPrecondition(
        "retained runtime closure differs from engine admission");
  }
  std::uint64_t capacity_template_exact_bytes = 0;
  for (const auto& descriptor : closure.descriptors()) {
    if (!(descriptor.object_root ==
          payload_root(admission.reference_roots_, descriptor.role))) {
      return Status::FailedPrecondition(
          "retained runtime descriptor differs from engine admission");
    }
    if (descriptor.role == RuntimeProfileReferenceRole::kCapacityTemplate) {
      capacity_template_exact_bytes = descriptor.exact_bytes;
    }
  }
  if (capacity_template_exact_bytes == 0) {
    return Status::FailedPrecondition(
        "runtime capacity template descriptor is absent");
  }
  auto status = verify_runtime_profile_reference_objects(closure, objects);
  if (!status.ok()) return status;
  admission.capacity_template_exact_bytes_ = capacity_template_exact_bytes;
  admission.reference_retention_ =
      RuntimeProfileReferenceRetention::kMemorySnapshot;
  admission.reference_anchor_ =
      std::make_shared<const std::vector<std::vector<std::byte>>>(
          std::move(objects));
  return admission;
}

}  // namespace pih
