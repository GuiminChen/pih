#pragma once
#include <filesystem>
#include <memory>
#include "pih/core/result.h"

namespace pih::offline_deepseek {
class GenerationStore final {
 public:
  // Opens the frozen store ABI with retained no-follow descriptors. Validates
  // structure/member kinds, not generation/receipt/pointer payload contents.
  static Result<std::unique_ptr<GenerationStore>> Open(const std::filesystem::path& path);
  ~GenerationStore();
  GenerationStore(const GenerationStore&) = delete;
  GenerationStore& operator=(const GenerationStore&) = delete;
  Status Revalidate() const;
  int root_descriptor() const noexcept;
  int staging_descriptor() const noexcept;
  int generations_descriptor() const noexcept;
  int receipts_descriptor() const noexcept;
  int pointers_descriptor() const noexcept;
 private:
  struct Impl;
  explicit GenerationStore(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};
// Exclusive creation of a previously absent root beneath an existing controlled
// parent. Failure retains partial creation for inspection; no recursive cleanup.
// Caller must exclude concurrent writers. No activation or model execution.
Status InitializeGenerationStore(const std::filesystem::path& path);
}  // namespace pih::offline_deepseek
