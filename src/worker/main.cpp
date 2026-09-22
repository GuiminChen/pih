#include <array>
#include <charconv>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <stdexcept>

#include "worker/worker_bootstrap.h"

namespace {

bool PrepareLoaderEnvironment() noexcept {
#if defined(__linux__)
  constexpr std::array loader_variables{
      "LD_PRELOAD",        "LD_AUDIT",        "LD_LIBRARY_PATH",
      "LD_DEBUG",          "LD_DEBUG_OUTPUT", "LD_PROFILE",
      "LD_PROFILE_OUTPUT", "LD_BIND_NOT",     "LD_DYNAMIC_WEAK",
      "LD_SHOW_AUXV",      "LD_ORIGIN_PATH",  "LD_HWCAP_MASK",
      "GLIBC_TUNABLES"};
  bool injected = false;
  for (const auto* variable : loader_variables) {
    const auto* value = std::getenv(variable);
    injected = injected || (value != nullptr && value[0] != '\0');
    if (::unsetenv(variable) != 0) return false;
  }
  return !injected;
#else
  return true;
#endif
}

}  // namespace

int main(int argc, char** argv) {
  if (!PrepareLoaderEnvironment()) {
    std::cerr << "pih_worker_failed: loader_environment_rejected\n";
    return 1;
  }
  const bool tokens_mode = argc == 7 && std::string(argv[3]) == "--generate-tokens" && std::string(argv[5]) == "--max-tokens";
  if ((argc == 3 || (argc == 5 && std::string(argv[3]) == "--serve") || tokens_mode) && std::string(argv[1]) == "--lock") {
    try {
      unsigned int port = 0;
      std::vector<std::uint32_t> tokens;
      std::uint32_t maximum = 128;
      if (tokens_mode) {
        const std::string input(argv[4]);
        if (input.empty() || input.size() > 512 * 1024) throw std::invalid_argument("prompt_tokens_invalid");
        std::size_t start = 0;
        while (start < input.size()) {
          auto end = input.find(',', start);
          if (end == std::string::npos) end = input.size();
          std::uint32_t token = 0;
          const auto parsed = std::from_chars(input.data() + start, input.data() + end, token);
          // Vocabulary validation belongs to the selected model capability.
          if (parsed.ec != std::errc{} || parsed.ptr != input.data() + end || tokens.size() >= 65536)
            throw std::invalid_argument("prompt_tokens_invalid");
          tokens.push_back(token);
          if (end + 1 == input.size()) throw std::invalid_argument("prompt_tokens_trailing_comma");
          start = end + 1;
        }
        const std::string limit(argv[6]);
        const auto parsed = std::from_chars(limit.data(), limit.data() + limit.size(), maximum);
        if (parsed.ec != std::errc{} || parsed.ptr != limit.data() + limit.size() || maximum == 0 || maximum > 65536)
          throw std::invalid_argument("maximum_tokens_invalid");
      }
      if (argc == 5) {
        const std::string text(argv[4]);
        const auto parsed = std::from_chars(text.data(), text.data() + text.size(), port);
        if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || port == 0 || port > 65535)
          throw std::invalid_argument("serve_port_invalid");
      }
      const auto lock_path =
          std::filesystem::absolute(std::filesystem::path(argv[2]))
              .lexically_normal()
              .string();
      return pih::worker::RunDevelopmentLock(lock_path, static_cast<std::uint16_t>(port), tokens, maximum);
    } catch (const std::exception& error) {
      std::cerr << "pih_worker_failed: " << error.what() << '\n';
      return 1;
    } catch (...) {
      std::cerr << "pih_worker_failed: nonstandard_exception\n";
      return 1;
    }
  }
  std::cerr << "usage: pih-worker --lock LOCK_PATH [--serve LOOPBACK_PORT | --generate-tokens ID,ID,... --max-tokens COUNT]\n";
  return 2;
}
