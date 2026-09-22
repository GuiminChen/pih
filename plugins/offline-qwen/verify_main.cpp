// Read-only CPU artifact verification. No model execution or publication.
#include "pih/io/mapped_file.h"
#include "pih/model/qwen3_int4_artifact_io.h"
#include <iostream>
#include <stdexcept>

namespace {
template<class T> T Require(pih::Result<T> result) {
  if (!result.ok()) throw std::runtime_error(std::string(result.status().message()));
  return std::move(*result);
}
pih::Sha256Digest Digest(const char* text) {
  const auto value = Require(pih::Sha256Digest::ParseHex(text));
  if (value == pih::Sha256Digest{} || value.hex() != text)
    throw std::invalid_argument("expected digest must be nonzero lowercase SHA-256");
  return value;
}
}

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cerr << "usage: pih-qwen-int4-verify FILE FILE_SHA256 SOURCE_ROOT BINDING_ROOT DISPOSITION_ROOT\n";
    return 2;
  }
  try {
    const auto file_digest = Digest(argv[2]);
    const auto source_root = Digest(argv[3]);
    const auto binding_root = Digest(argv[4]);
    const auto disposition_root = Digest(argv[5]);
    auto file = Require(pih::MappedFile::OpenReadOnly(
        argv[1], pih::QwenInt4ArtifactLayout::kOfficialFileBytes));
    if (file.size_bytes() != pih::QwenInt4ArtifactLayout::kOfficialFileBytes)
      throw std::invalid_argument("Qwen INT4 artifact length mismatch");
    const auto verified = Require(pih::verify_qwen_int4_artifact(
        std::span<const std::byte>(file.data(), static_cast<size_t>(file.size_bytes()))));
    if (verified.file.file_sha256 != file_digest ||
        verified.roots.source_artifact != source_root ||
        verified.roots.source_binding != binding_root ||
        verified.roots.disposition != disposition_root)
      throw std::invalid_argument("artifact differs from expected file or source bindings");
    std::cout << "{\"schema\":\"pih.qwen3_int4_verification_receipt.v1\",\"file_bytes\":"
              << verified.file.file_bytes << ",\"file_sha256\":\"" << file_digest.hex()
              << "\",\"source_artifact_root\":\"" << source_root.hex()
              << "\",\"source_binding_root\":\"" << binding_root.hex()
              << "\",\"disposition_root\":\"" << disposition_root.hex() << "\"}\n";
    std::cout.flush();
    if (!std::cout) throw std::runtime_error("verification receipt output failed");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "native Qwen artifact verification failed: " << error.what() << '\n';
    return 2;
  }
}
