// Read-only BF16 source validation, without conversion or model execution.
#include "source_expectation.h"
#include <iostream>

namespace {
template<class T> T Require(pih::Result<T> result) {
  if (!result.ok()) throw std::runtime_error(std::string(result.status().message()));
  return std::move(*result);
}
}
int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: pih-qwen-source-verify SOURCE.safetensors EXPECTATION.json\n";
    return 2;
  }
  try {
    const auto expected = pih::qwen_offline::ReadSourceExpectation(argv[2]);
    const auto source = Require(pih::SafetensorsFile::Open(argv[1], pih::qwen_offline::kSourceByteLimit));
    const auto valid = pih::Qwen3Manifest::Validate(source.header());
    if (!valid.ok()) throw std::invalid_argument(std::string(valid.message()));
    const auto receipt = Require(pih::verify_qwen3_source_artifact(source, expected));
    std::cout << "{\"schema\":\"pih.qwen3_source_verification_receipt.v1\",\"file_bytes\":"
              << receipt.file_bytes << ",\"tensor_count\":" << receipt.tensor_count
              << ",\"data_bytes\":" << receipt.data_bytes << ",\"file_sha256\":\""
              << receipt.file_sha256.hex() << "\",\"tied_weight_bytes\":" << receipt.tied_weight_bytes
              << ",\"embedding_sha256\":\"" << receipt.embedding_sha256.hex()
              << "\",\"lm_head_sha256\":\"" << receipt.lm_head_sha256.hex() << "\"}\n";
    std::cout.flush();
    if (!std::cout) throw std::runtime_error("source verification receipt output failed");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "native Qwen source verification failed: " << error.what() << '\n';
    return 2;
  }
}
