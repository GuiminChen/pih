#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

#include "worker/deployment_lock.h"

namespace {

void WriteJsonString(std::ostream& output, std::string_view value) {
  constexpr char hex[] = "0123456789abcdef";
  output << '"';
  for (const unsigned char byte : value) {
    switch (byte) {
      case '"': output << "\\\""; break;
      case '\\': output << "\\\\"; break;
      case '\b': output << "\\b"; break;
      case '\f': output << "\\f"; break;
      case '\n': output << "\\n"; break;
      case '\r': output << "\\r"; break;
      case '\t': output << "\\t"; break;
      default:
        if (byte < 0x20) {
          output << "\\u00" << hex[byte >> 4] << hex[byte & 0x0f];
        } else {
          output << static_cast<char>(byte);
        }
    }
  }
  output << '"';
}

void Inspect(const std::string& lock_path) {
  const auto lock = pih::worker::LoadDevelopmentLock(lock_path);
  std::cout << "{\"schema\":\"pih.inspect.v1\",\"support_status\":"
               "\"unsupported\",\"deployment_root\":";
  WriteJsonString(std::cout, lock.deployment_root);
  std::cout << ",\"kernel_packs\":[";
  for (std::size_t index = 0; index < lock.kernel_packs.size(); ++index) {
    const auto& pack = lock.kernel_packs[index];
    if (index != 0) std::cout << ',';
    std::cout << "{\"pack_id\":";
    WriteJsonString(std::cout, pack.pack_id);
    std::cout << ",\"pack_version\":";
    WriteJsonString(std::cout, pack.pack_version);
    std::cout << ",\"pack_abi\":";
    WriteJsonString(std::cout, pack.pack_abi);
    std::cout << ",\"architecture\":";
    WriteJsonString(std::cout, pack.architecture);
    std::cout << ",\"binary\":";
    WriteJsonString(std::cout, pack.binary);
    if (!pack.binary_sha256_hex.empty()) {
      std::cout << ",\"binary_sha256_hex\":";
      WriteJsonString(std::cout, pack.binary_sha256_hex);
    }
    std::cout << '}';
  }
  std::cout << "],\"plugins\":[";
  for (std::size_t index = 0; index < lock.plugins.size(); ++index) {
    const auto& plugin = lock.plugins[index];
    if (index != 0) std::cout << ',';
    std::cout << "{\"plugin_id\":";
    WriteJsonString(std::cout, plugin.plugin_id);
    std::cout << ",\"plugin_version\":";
    WriteJsonString(std::cout, plugin.plugin_version);
    std::cout << ",\"entrypoint\":";
    WriteJsonString(std::cout, plugin.entrypoint);
    if (!plugin.entrypoint_sha256_hex.empty()) {
      std::cout << ",\"entrypoint_sha256_hex\":";
      WriteJsonString(std::cout, plugin.entrypoint_sha256_hex);
    }
    std::cout << '}';
  }
  std::cout << "],\"capabilities\":[";
  for (std::size_t index = 0; index < lock.capabilities.size(); ++index) {
    const auto& capability = lock.capabilities[index];
    if (index != 0) std::cout << ',';
    std::cout << "{\"capability_id\":";
    WriteJsonString(std::cout, capability.capability_id);
    std::cout << ",\"provider_id\":";
    WriteJsonString(std::cout, capability.provider_id);
    std::cout << ",\"contract_id\":";
    WriteJsonString(std::cout, capability.contract_id);
    std::cout << ",\"threading_model\":" << capability.threading_model
              << ",\"scope\":" << capability.scope
              << ",\"cardinality\":" << capability.cardinality << '}';
  }
  std::cout << "],\"engine\":";
  if (!lock.has_engine) {
    std::cout << "null";
  } else {
    std::cout << "{\"capability_id\":";
    WriteJsonString(std::cout, lock.engine.capability_id);
    std::cout << ",\"contract_id\":";
    WriteJsonString(std::cout, lock.engine.contract_id);
    std::cout << ",\"activation_epoch\":"
              << lock.engine.activation_epoch
              << ",\"configuration\":"
              << lock.engine.configuration_json << '}';
  }
  std::cout << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4 || std::string_view(argv[1]) != "inspect" ||
      std::string_view(argv[2]) != "--lock") {
    std::cerr << "usage: pih inspect --lock LOCK_PATH\n";
    return 2;
  }
  try {
    const auto lock_path =
        std::filesystem::absolute(std::filesystem::path(argv[3]))
            .lexically_normal()
            .string();
    Inspect(lock_path);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "pih_inspect_failed: " << error.what() << '\n';
    return 1;
  } catch (...) {
    std::cerr << "pih_inspect_failed: nonstandard_exception\n";
    return 1;
  }
}
