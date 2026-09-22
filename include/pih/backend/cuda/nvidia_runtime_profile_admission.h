#pragma once

#include <cstddef>
#include <array>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include "pih/model/runtime_profile_payload.h"
#include "pih/model/runtime_profile_supervisor_bootstrap_manifest.h"

namespace pih {

Result<RuntimeEngineAdmission> admit_nvidia_runtime_profile(
    std::span<const std::byte> trust_policy_bytes,
    std::span<const std::byte> catalog_bytes,
    std::span<const std::byte> envelope_bytes,
    std::span<const std::byte> reference_closure_bytes,
    std::span<const std::vector<std::byte>> ordered_reference_objects,
    std::string_view profile_id,
    std::string_view revision,
    std::span<const std::int32_t> ordered_device_ordinals,
    RuntimeProfileModel expected_model,
    RuntimeProfileWeightFormat expected_weight_format);

#if defined(__linux__)
Result<RuntimeEngineAdmission> admit_nvidia_runtime_profile_from_leases(
    std::span<const std::byte> trust_policy_bytes,
    std::span<const std::byte> catalog_bytes,
    std::span<const std::byte> envelope_bytes,
    std::span<const std::byte> reference_closure_bytes,
    std::span<const std::byte> graph_manifest_bytes,
    const RuntimeProfileSupervisorBootstrapManifest& bootstrap_manifest,
    std::string_view profile_id,
    std::string_view revision,
    std::span<const std::int32_t> ordered_device_ordinals,
    RuntimeProfileModel expected_model,
    RuntimeProfileWeightFormat expected_weight_format);
#endif

}  // namespace pih
