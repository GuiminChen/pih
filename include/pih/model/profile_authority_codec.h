#pragma once

#include <span>

#include "pih/model/profile_authority.h"

namespace pih {

Result<ProfileTrustPolicy> parse_profile_trust_policy(
    std::span<const std::byte> bytes);
Result<SignedProfileCatalog> parse_signed_profile_catalog(
    std::span<const std::byte> bytes);
Result<SignedProfileEnvelope> parse_signed_profile_envelope(
    std::span<const std::byte> bytes);

}  // namespace pih
