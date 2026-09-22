#include "pih/backend/cuda/qwen_int4_cuda_fixture.h"

#include <exception>
#include <iostream>
#include <string_view>

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: pih-qwen-cuda-qualify shape|gemm|metric\n"
              << "set PIH_QWEN_CUBIN_PATH to the absolute deployed cubin path\n";
    return 2;
  }
  try {
    const std::string_view command(argv[1]);
    if (command == "shape") {
      auto result = pih::cuda_collect_qwen_int4_gemm_shape_matrix();
      if (!result.ok()) {
        std::cerr << "Qwen shape qualification failed: " << result.status().message() << '\n';
        return 1;
      }
      std::cout << "{\"cases\":[";
      for (std::size_t index = 0; index < result->cases.size(); ++index) {
        if (index) std::cout << ',';
        const auto& item = result->cases[index];
        std::cout << "{\"m\":" << item.m << ",\"n\":" << item.n
                  << ",\"k\":" << item.k << ",\"pattern_seed\":" << item.pattern_seed
                  << ",\"output_sha256\":\"" << item.output_sha256 << "\"}";
      }
      std::cout << "],\"negative_launch_invariants\":["
                << result->negative_launch_invariants[0] << ','
                << result->negative_launch_invariants[1] << ','
                << result->negative_launch_invariants[2] << "]}\n";
      return std::cout ? 0 : 1;
    }
    pih::Status status = command == "gemm"
        ? pih::cuda_verify_qwen_int4_gemm_fixture()
        : command == "metric"
            ? pih::cuda_verify_qwen_teacher_forced_metric_fixture()
            : pih::Status::InvalidArgument("unknown qualification command");
    if (!status.ok()) {
      std::cerr << "Qwen CUDA qualification failed: " << status.message() << '\n';
      return command == "gemm" || command == "metric" ? 1 : 2;
    }
    std::cout << "{\"schema\":\"pih.qwen_cuda_qualification.v1\",\"status\":\"passed\"}\n";
    return std::cout ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "Qwen CUDA qualification failed: " << error.what() << '\n';
    return 1;
  }
}
