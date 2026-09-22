// Native CPU-only artifact tool. No Python runtime, CUDA or monolith linkage.
#include "source_expectation.h"
#include "qwen3_int4_converter.h"
#include "pih/model/qwen3_manifest.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cfenv>

namespace {
template<class T> T Require(pih::Result<T> result) {
  if (!result.ok()) throw std::runtime_error(std::string(result.status().message()));
  return std::move(*result);
}
}
int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: pih-qwen-int4-convert SOURCE.safetensors EXPECTATION.json OUTPUT.xing-int4\n";
    return 2;
  }
  bool published = false;
  try {
    if (std::fegetround() != FE_TONEAREST)
      throw std::invalid_argument("conversion requires round-to-nearest floating-point mode");
    const auto source_path = std::filesystem::canonical(argv[1]);
    const auto expectation_path = std::filesystem::canonical(argv[2]);
    const auto output = std::filesystem::absolute(argv[3]).lexically_normal();
    const auto staging = std::filesystem::path(output.string() + ".staging");
    if (source_path == output || std::filesystem::exists(output) || std::filesystem::exists(staging))
      throw std::invalid_argument("output and staging must not exist");
    const auto expectation = pih::qwen_offline::ReadSourceExpectation(expectation_path);
    auto source = Require(pih::SafetensorsFile::Open(source_path, pih::qwen_offline::kSourceByteLimit));
    const auto validation = pih::Qwen3Manifest::Validate(source.header());
    if (!validation.ok()) throw std::invalid_argument(std::string(validation.message()));
    const auto receipt = Require(pih::verify_qwen3_source_artifact(source, expectation));
    const auto binding = Require(pih::QwenInt4SourceBinding::ObserveAndVerifyOfficial(source, receipt));
    const auto converted = Require(pih::convert_qwen3_official_to_int4(source, receipt, binding, staging, output, 1048576));
    published = true;
    std::cout << "{\"schema\":\"pih.qwen3_int4_conversion_receipt.v1\",\"file_bytes\":"
      << converted.publication.file_bytes << ",\"file_sha256\":\"" << converted.publication.file_sha256.hex()
      << "\",\"source_artifact_root\":\"" << converted.roots.source_artifact.hex()
      << "\",\"source_binding_root\":\"" << converted.roots.source_binding.hex()
      << "\",\"disposition_root\":\"" << converted.roots.disposition.hex() << "\"}\n";
    std::cout.flush();
    if (!std::cout) throw std::runtime_error("conversion receipt output failed");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "native Qwen conversion failed: " << error.what() << '\n';
    if (published) std::cerr << "artifact was published; receipt delivery failed; do not overwrite or blindly rerun\n";
    else std::cerr << "inspect output and .staging before retry; failure does not prove publication was absent\n";
    return 2;
  }
}
