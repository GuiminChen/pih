#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>

#include "pih/io/canonical_extent_writer.h"

namespace pih {

class DurableFileSink final : public CanonicalSequentialSink {
 public:
  static Result<DurableFileSink> CreateExclusive(
      const std::filesystem::path& path, std::uint64_t expected_file_bytes);

  DurableFileSink(const DurableFileSink&) = delete;
  DurableFileSink& operator=(const DurableFileSink&) = delete;
  DurableFileSink(DurableFileSink&& other) noexcept;
  DurableFileSink& operator=(DurableFileSink&& other) noexcept;
  ~DurableFileSink() override;

  Result<std::size_t> write(std::span<const std::byte> input) override;
  Status sync() override;
  [[nodiscard]] std::uint64_t bytes_written() const noexcept {
    return bytes_written_;
  }

 private:
  DurableFileSink(int descriptor, std::uint64_t expected_file_bytes)
      : descriptor_(descriptor), expected_file_bytes_(expected_file_bytes) {}
  void close() noexcept;

  int descriptor_ = -1;
  std::uint64_t expected_file_bytes_ = 0;
  std::uint64_t bytes_written_ = 0;
  bool synced_ = false;
};

}  // namespace pih
