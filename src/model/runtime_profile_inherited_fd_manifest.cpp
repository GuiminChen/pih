#include "pih/model/runtime_profile_inherited_fd_manifest.h"

#include <string_view>

namespace pih {
namespace {

void append_u32(std::vector<std::byte>& out, std::uint32_t value) {
  out.push_back(static_cast<std::byte>((value >> 24) & 0xff));
  out.push_back(static_cast<std::byte>((value >> 16) & 0xff));
  out.push_back(static_cast<std::byte>((value >> 8) & 0xff));
  out.push_back(static_cast<std::byte>(value & 0xff));
}

template <std::size_t Size>
bool valid_distinct_fds(const std::array<std::int32_t, Size>& values) {
  for (std::size_t left = 0; left < values.size(); ++left) {
    if (values[left] < 3) return false;
    for (std::size_t right = left + 1; right < values.size(); ++right) {
      if (values[left] == values[right]) return false;
    }
  }
  return true;
}

}  // namespace

RuntimeProfileInheritedFdManifest::RuntimeProfileInheritedFdManifest(
    std::array<std::int32_t, 5> authority_fds,
    std::array<std::int32_t, 7> reference_fds,
    Sha256Digest manifest_root,
    std::vector<std::byte> canonical_bytes) noexcept
    : authority_fds_(authority_fds), reference_fds_(reference_fds),
      manifest_root_(manifest_root), canonical_bytes_(std::move(canonical_bytes)) {}

Result<RuntimeProfileInheritedFdManifest>
RuntimeProfileInheritedFdManifest::Create(
    std::array<std::int32_t, 5> authority_fds,
    std::array<std::int32_t, 7> reference_fds) {
  if (!valid_distinct_fds(authority_fds) ||
      !valid_distinct_fds(reference_fds)) {
    return Status::InvalidArgument(
        "runtime profile inherited fd manifest contains invalid or duplicate fd");
  }
  for (const auto authority_fd : authority_fds) {
    for (const auto reference_fd : reference_fds) {
      if (authority_fd == reference_fd) {
        return Status::InvalidArgument(
            "runtime profile inherited fd manifest aliases roles");
      }
    }
  }

  constexpr std::string_view abi = "runtime_profile_inherited_fd_manifest_v2";
  std::vector<std::byte> canonical;
  canonical.reserve(abi.size() + 2 + 12 * sizeof(std::uint32_t));
  const auto abi_bytes = std::as_bytes(std::span(abi.data(), abi.size()));
  canonical.insert(canonical.end(), abi_bytes.begin(), abi_bytes.end());
  canonical.push_back(std::byte{5});
  for (const auto fd : authority_fds) {
    append_u32(canonical, static_cast<std::uint32_t>(fd));
  }
  canonical.push_back(std::byte{7});
  for (const auto fd : reference_fds) {
    append_u32(canonical, static_cast<std::uint32_t>(fd));
  }
  auto root = sha256(canonical);
  if (!root.ok()) return root.status();
  return RuntimeProfileInheritedFdManifest(
      authority_fds, reference_fds, *root, std::move(canonical));
}

}  // namespace pih
