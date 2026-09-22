#pragma once
#include "weight_source_wo_a.h"
#include "pih/core/sha256.h"
#include <memory>

namespace pih::deepseek_v41 {
// Linux-owned read-only source shard. Digest/size must come from an independent
// trusted source manifest. Does not establish a complete checkpoint inventory.
class WeightSourceFile final {
 public:
  static Result<std::unique_ptr<WeightSourceFile>> Open(int source_directory_fd,
      std::string_view member, std::uint64_t bytes, const Sha256Digest& digest);
  ~WeightSourceFile();
  WeightSourceFile(const WeightSourceFile&) = delete;
  WeightSourceFile& operator=(const WeightSourceFile&) = delete;
  const SafetensorsHeader& header() const noexcept;
  std::string_view member_name() const noexcept;
  Status Revalidate() const;
  // Exact file-relative reads, 1..1 MiB, checked against the admitted size.
  // Checks descriptor and directory-entry identity before and after each read.
  Status Read(std::uint64_t offset, std::span<std::byte> bytes) const;
 private:
  struct Impl;
  explicit WeightSourceFile(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};
Status ConvertWoASourceFiles(std::uint32_t layer,
    const WeightSourceFile& weight_file, std::string_view weight_name,
    const WeightSourceFile& scale_file, std::string_view scale_name,
    const WoATensorWriter& write);
}  // namespace pih::deepseek_v41
