#pragma once

#include <string>
#include <string_view>

namespace pih::microkernel {

class KernelPackInstance final {
 public:
  KernelPackInstance(void* native_handle, std::string id, std::string version,
                     std::string pack_abi, std::string architecture,
                     const void* contract_api);
  KernelPackInstance(const KernelPackInstance&) = delete;
  KernelPackInstance& operator=(const KernelPackInstance&) = delete;
  KernelPackInstance(KernelPackInstance&& other) noexcept;
  KernelPackInstance& operator=(KernelPackInstance&&) = delete;
  ~KernelPackInstance() = default;

  [[nodiscard]] const std::string& id() const noexcept { return id_; }
  [[nodiscard]] const std::string& version() const noexcept { return version_; }
  [[nodiscard]] const std::string& pack_abi() const noexcept {
    return pack_abi_;
  }
  [[nodiscard]] const std::string& architecture() const noexcept {
    return architecture_;
  }
  [[nodiscard]] const void* contract_api() const noexcept {
    return contract_api_;
  }

 private:
  void* native_handle_{};
  std::string id_;
  std::string version_;
  std::string pack_abi_;
  std::string architecture_;
  const void* contract_api_{};
};

class KernelPackLoader final {
 public:
  KernelPackLoader() = default;
  KernelPackLoader(const KernelPackLoader&) = delete;
  KernelPackLoader& operator=(const KernelPackLoader&) = delete;

  [[nodiscard]] KernelPackInstance Load(
      const std::string& path,
      std::string_view expected_sha256_hex = {});
  void Seal();
  [[nodiscard]] bool sealed() const noexcept { return sealed_; }

 private:
  bool sealed_{};
};

}  // namespace pih::microkernel
