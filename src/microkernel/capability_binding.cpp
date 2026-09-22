#include "microkernel/capability_binding.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>

#include "pih/plugin_sdk/capability.h"

namespace pih::microkernel {
namespace {

constexpr std::size_t kMaximumCapabilityIdentifierBytes = 256;

bool ValidId(const std::string& value) {
  if (value.empty() || value.size() > kMaximumCapabilityIdentifierBytes ||
      value.front() == '.' || value.back() == '.' ||
      value.find("..") != std::string::npos) {
    return false;
  }
  for (const char byte : value) {
    if (!((byte >= 'a' && byte <= 'z') || (byte >= '0' && byte <= '9') ||
          byte == '.' || byte == '-')) {
      return false;
    }
  }
  return true;
}

}  // namespace

CapabilityBindings::CapabilityBindings(std::uint64_t activation_epoch)
    : activation_epoch_(activation_epoch) {
  if (activation_epoch_ == 0) {
    throw std::invalid_argument("capability_activation_epoch_invalid");
  }
}

void CapabilityBindings::Bind(std::string capability_id,
                              std::string provider_id,
                              std::string contract_id, const void* api,
                              std::uint32_t threading_model, std::uint32_t scope,
                              std::uint32_t cardinality) {
  if (sealed_ || revoked_) {
    throw std::logic_error("capability_bindings_sealed");
  }
  if (!ValidId(capability_id) || !ValidId(provider_id) ||
      !ValidId(contract_id) || api == nullptr ||
      threading_model < PIH_CAPABILITY_THREADING_SINGLE_THREADED_V1 ||
      threading_model > PIH_CAPABILITY_THREADING_CONCURRENT_V1 ||
      scope < PIH_CAPABILITY_SCOPE_PROCESS_V1 ||
      scope > PIH_CAPABILITY_SCOPE_REQUEST_V1 ||
      cardinality != PIH_CAPABILITY_CARDINALITY_EXACTLY_ONE_V1 ||
      !bindings_
           .emplace(std::move(capability_id),
                    Binding{std::move(provider_id), std::move(contract_id), api,
                            activation_epoch_, threading_model, scope,
                            cardinality})
           .second) {
    throw std::invalid_argument("capability_binding_invalid");
  }
}

void CapabilityBindings::Seal() {
  if (sealed_ || revoked_) {
    throw std::logic_error("capability_bindings_not_sealable");
  }
  sealed_ = true;
}

void CapabilityBindings::Revoke() {
  if (!sealed_ || revoked_) {
    throw std::logic_error("capability_bindings_not_revocable");
  }
  bindings_.clear();
  revoked_ = true;
}

const std::string& CapabilityBindings::Resolve(
    const std::string& capability_id) const {
  if (!sealed_ || revoked_) {
    throw std::logic_error("capability_bindings_not_resolvable");
  }
  const auto found = bindings_.find(capability_id);
  if (found == bindings_.end() ||
      found->second.activation_epoch != activation_epoch_) {
    throw std::out_of_range("capability_missing");
  }
  return found->second.provider_id;
}

const void* CapabilityBindings::ResolveApi(
    const std::string& capability_id, const std::string& contract_id,
    std::uint32_t required_scope,
    std::uint32_t required_cardinality) const {
  if (!sealed_ || revoked_) {
    throw std::logic_error("capability_bindings_not_resolvable");
  }
  if (!ValidId(capability_id) || !ValidId(contract_id) ||
      required_scope < PIH_CAPABILITY_SCOPE_PROCESS_V1 ||
      required_scope > PIH_CAPABILITY_SCOPE_REQUEST_V1 ||
      (required_cardinality !=
           PIH_CAPABILITY_CARDINALITY_EXACTLY_ONE_V1 &&
       required_cardinality !=
           PIH_CAPABILITY_CARDINALITY_ZERO_OR_ONE_V1)) {
    throw std::invalid_argument("capability_requirement_invalid");
  }
  const auto found = bindings_.find(capability_id);
  if (found == bindings_.end()) {
    if (required_cardinality ==
        PIH_CAPABILITY_CARDINALITY_ZERO_OR_ONE_V1) {
      return nullptr;
    }
    throw std::out_of_range("capability_contract_missing");
  }
  if (found->second.activation_epoch != activation_epoch_ ||
      found->second.contract_id != contract_id ||
      found->second.api == nullptr ||
      found->second.scope > required_scope ||
      found->second.cardinality !=
          PIH_CAPABILITY_CARDINALITY_EXACTLY_ONE_V1) {
    throw std::out_of_range("capability_contract_missing");
  }
  return found->second.api;
}

std::vector<CapabilityDescription> CapabilityBindings::Describe() const {
  if (!sealed_ || revoked_) {
    throw std::logic_error("capability_bindings_not_describable");
  }
  std::vector<CapabilityDescription> result;
  result.reserve(bindings_.size());
  for (const auto& [capability_id, binding] : bindings_) {
    if (binding.activation_epoch != activation_epoch_) {
      throw std::logic_error("capability_binding_epoch_drifted");
    }
    result.push_back(
        {capability_id, binding.provider_id, binding.contract_id,
         binding.threading_model, binding.scope, binding.cardinality});
  }
  std::sort(result.begin(), result.end(),
            [](const auto& left, const auto& right) {
              return left.capability_id < right.capability_id;
            });
  return result;
}

}  // namespace pih::microkernel
