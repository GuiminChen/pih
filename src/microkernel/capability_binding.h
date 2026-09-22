#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace pih::microkernel {

struct CapabilityDescription final {
  std::string capability_id;
  std::string provider_id;
  std::string contract_id;
  std::uint32_t threading_model{};
  std::uint32_t scope{};
  std::uint32_t cardinality{};
};

class CapabilityBindings final {
 public:
  explicit CapabilityBindings(std::uint64_t activation_epoch);
  CapabilityBindings(const CapabilityBindings&) = delete;
  CapabilityBindings& operator=(const CapabilityBindings&) = delete;
  void Bind(std::string capability_id, std::string provider_id,
            std::string contract_id, const void* api,
            std::uint32_t threading_model, std::uint32_t scope,
            std::uint32_t cardinality);
  void Seal();
  void Revoke();
  [[nodiscard]] bool sealed() const noexcept { return sealed_; }
  [[nodiscard]] bool revoked() const noexcept { return revoked_; }
  [[nodiscard]] std::uint64_t activation_epoch() const noexcept {
    return activation_epoch_;
  }
  [[nodiscard]] const std::string& Resolve(const std::string& capability_id) const;
  [[nodiscard]] const void* ResolveApi(const std::string& capability_id,
                                       const std::string& contract_id,
                                       std::uint32_t required_scope,
                                       std::uint32_t required_cardinality) const;
  [[nodiscard]] std::vector<CapabilityDescription> Describe() const;

 private:
  struct Binding final {
    std::string provider_id;
    std::string contract_id;
    const void* api{};
    std::uint64_t activation_epoch{};
    std::uint32_t threading_model{};
    std::uint32_t scope{};
    std::uint32_t cardinality{};
  };
  std::unordered_map<std::string, Binding> bindings_;
  std::uint64_t activation_epoch_{};
  bool sealed_{};
  bool revoked_{};
};

}  // namespace pih::microkernel
