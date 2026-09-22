#include "pih/backend/cuda/qwen_int4_cuda_fixture.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <vector>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include "pih/backend/cuda/cuda_driver_status.h"
#include "pih/backend/cuda/cuda_status.h"
#include "pih/core/result.h"
#include "pih/core/sha256.h"
#include "pih/model/qwen3_cuda_invariant.h"
#include "pih/model/qwen3_teacher_forced_metric_oracle.h"

namespace pih {
namespace {

std::uint16_t bf16(float value) {
  std::uint32_t raw = 0;
  std::memcpy(&raw, &value, sizeof(raw));
  const std::uint32_t magnitude = raw & 0x7fffffffU;
  if (magnitude > 0x7f800000U)
    return static_cast<std::uint16_t>((raw >> 16U) | 0x0040U);
  return static_cast<std::uint16_t>(
      (raw + 0x7fffU + ((raw >> 16U) & 1U)) >> 16U);
}

void store_nibble(std::vector<std::uint8_t>& packed, std::uint64_t row,
                  std::uint64_t column, std::uint64_t packed_columns,
                  int value) {
  const auto nibble = static_cast<std::uint8_t>(value) & 0x0fU;
  const auto index = row * packed_columns + column / 2U;
  const auto shift = (column & 1U) == 0 ? 0U : 4U;
  packed[index] = static_cast<std::uint8_t>(packed[index] | (nibble << shift));
}

struct FixtureDeviceMemory final {
  std::uint16_t* input = nullptr;
  std::uint8_t* packed = nullptr;
  std::uint16_t* scales = nullptr;
  std::uint16_t* output = nullptr;
  std::uint32_t* error = nullptr;
  cudaStream_t stream = nullptr;
  CUmodule module = nullptr;
  ~FixtureDeviceMemory() {
    if (stream != nullptr) cudaStreamDestroy(stream);
    if (module != nullptr) cuModuleUnload(module);
    if (error != nullptr) cudaFree(error);
    if (output != nullptr) cudaFree(output);
    if (scales != nullptr) cudaFree(scales);
    if (packed != nullptr) cudaFree(packed);
    if (input != nullptr) cudaFree(input);
  }
};

struct MetricDeviceMemory final {
  float* logits = nullptr;
  std::uint32_t* targets = nullptr;
  std::uint32_t* argmax = nullptr;
  double* nll = nullptr;
  std::uint32_t* nonfinite = nullptr;
  std::uint32_t* error = nullptr;
  cudaStream_t stream = nullptr;
  CUmodule module = nullptr;
  ~MetricDeviceMemory() {
    if (stream != nullptr) cudaStreamDestroy(stream);
    if (module != nullptr) cuModuleUnload(module);
    if (error != nullptr) cudaFree(error);
    if (nonfinite != nullptr) cudaFree(nonfinite);
    if (nll != nullptr) cudaFree(nll);
    if (argmax != nullptr) cudaFree(argmax);
    if (targets != nullptr) cudaFree(targets);
    if (logits != nullptr) cudaFree(logits);
  }
};

Status runtime(cudaError_t result, const char* operation) {
  return result == cudaSuccess ? Status::Ok() : cuda_status(result, operation);
}

Status verify_case(const std::filesystem::path& cubin_path, std::uint64_t m,
                   std::uint64_t n, std::uint64_t k,
                   bool verify_reserved_nibble,
                   bool verify_invalid_launches, std::uint64_t pattern_seed,
                   std::string* output_sha256) {
  const std::uint64_t packed_columns = k / 2, groups = k / 128;
  constexpr std::array<float, 7> row_pattern{
      1.0F, -0.5F, 2.0F, -1.5F, 0.25F, 0.75F, -2.0F};
  const std::array<std::uint64_t, 6> positions{0, 1, 127, 128, 129, k - 1};
  constexpr std::array<int, 6> values{-7, -1, 1, 7, -7, 1};
  std::vector<std::uint16_t> input(m * k), scales(n * groups);
  std::vector<std::uint8_t> packed(n * packed_columns, 0);
  std::vector<std::uint16_t> expected(m * n), observed(m * n, 0);
  for (std::uint64_t row = 0; row < m; ++row)
    for (std::uint64_t inner = 0; inner < k; ++inner)
      input[row * k + inner] = bf16(row_pattern[row % row_pattern.size()]);
  for (std::uint64_t output = 0; output < n; ++output) {
    const int direction = (output & 1U) == 0 ? 1 : -1;
    float sum = 0.0F;
    for (std::uint64_t group = 0; group < groups; ++group) {
      const float scale = pattern_seed == 0
                              ? ((group & 1U) == 0 ? 1.0F : 0.5F)
                              : std::array<float, 4>{0.25F, 0.5F, 1.0F, 2.0F}
                                    [(group + output + pattern_seed) & 3U];
      scales[output * groups + group] =
          __half_as_ushort(__float2half_rn(scale));
    }
    if (pattern_seed == 0) {
      for (std::size_t index = 0; index < positions.size(); ++index) {
        const int value = values[index] * direction;
        store_nibble(packed, output, positions[index], packed_columns, value);
        const float scale = ((positions[index] / 128U) & 1U) == 0 ? 1.0F : 0.5F;
        sum += static_cast<float>(value) * scale;
      }
    } else {
      for (std::uint64_t inner = 0; inner < k; ++inner) {
        std::uint64_t mixed = pattern_seed ^ (output + 1U) * 0x9e3779b97f4a7c15ULL;
        mixed ^= (inner + 1U) * 0xbf58476d1ce4e5b9ULL;
        mixed ^= mixed >> 30U;
        mixed *= 0xbf58476d1ce4e5b9ULL;
        mixed ^= mixed >> 27U;
        const int value = static_cast<int>(mixed % 15U) - 7;
        store_nibble(packed, output, inner, packed_columns, value);
        const auto group = inner / 128U;
        const float scale =
            std::array<float, 4>{0.25F, 0.5F, 1.0F, 2.0F}
                [(group + output + pattern_seed) & 3U];
        sum += static_cast<float>(value) * scale;
      }
    }
    for (std::uint64_t row = 0; row < m; ++row)
      expected[row * n + output] =
          bf16(row_pattern[row % row_pattern.size()] * sum);
  }

  FixtureDeviceMemory device;
  Status status = runtime(cudaMalloc(reinterpret_cast<void**>(&device.input),
                                     input.size() * sizeof(input[0])),
                          "Qwen fixture cudaMalloc input");
  if (status.ok()) status = runtime(cudaMalloc(reinterpret_cast<void**>(&device.packed), packed.size()), "Qwen fixture cudaMalloc packed");
  if (status.ok()) status = runtime(cudaMalloc(reinterpret_cast<void**>(&device.scales), scales.size() * sizeof(scales[0])), "Qwen fixture cudaMalloc scales");
  if (status.ok()) status = runtime(cudaMalloc(reinterpret_cast<void**>(&device.output), observed.size() * sizeof(observed[0])), "Qwen fixture cudaMalloc output");
  if (status.ok()) status = runtime(cudaMalloc(reinterpret_cast<void**>(&device.error), sizeof(std::uint32_t)), "Qwen fixture cudaMalloc error");
  if (status.ok()) status = runtime(cudaStreamCreateWithFlags(&device.stream, cudaStreamNonBlocking), "Qwen fixture stream create");
  if (!status.ok()) return status;
  CUresult driver = cuModuleLoad(&device.module, cubin_path.string().c_str());
  if (driver != CUDA_SUCCESS) return cuda_driver_status(driver, "cuModuleLoad Qwen fixture");
  CUfunction function = nullptr;
  driver = cuModuleGetFunction(&function, device.module,
                               "pih_qwen_w4a16_gemm_compat_v1");
  if (driver != CUDA_SUCCESS) return cuda_driver_status(driver, "cuModuleGetFunction Qwen fixture");

  status = runtime(
      cudaMemcpyAsync(device.input, input.data(),
                      input.size() * sizeof(input[0]), cudaMemcpyHostToDevice,
                      device.stream),
      "Qwen fixture input copy");
  if (status.ok())
    status = runtime(cudaMemcpyAsync(device.packed, packed.data(), packed.size(),
                                     cudaMemcpyHostToDevice, device.stream),
                     "Qwen fixture packed copy");
  if (status.ok())
    status = runtime(
        cudaMemcpyAsync(device.scales, scales.data(),
                        scales.size() * sizeof(scales[0]),
                        cudaMemcpyHostToDevice, device.stream),
        "Qwen fixture scale copy");
  if (status.ok())
    status = runtime(
        cudaMemsetAsync(device.error, 0, sizeof(std::uint32_t), device.stream),
        "Qwen fixture error clear");
  if (!status.ok()) return status;
  std::uint64_t launch_m = m;
  std::uint64_t launch_n = n;
  std::uint64_t launch_k = k;
  void* arguments[] = {&device.input, &device.packed, &device.scales,
                       &device.output, &device.error, &launch_m, &launch_n,
                       &launch_k};
  const auto blocks = static_cast<unsigned int>((m * n + 255U) / 256U);
  driver = cuLaunchKernel(function, blocks, 1, 1, 256, 1, 1, 0,
                          reinterpret_cast<CUstream>(device.stream), arguments,
                          nullptr);
  if (driver != CUDA_SUCCESS) return cuda_driver_status(driver, "cuLaunchKernel Qwen fixture");
  std::uint32_t error_code = 0;
  status = runtime(cudaMemcpyAsync(observed.data(), device.output, observed.size() * sizeof(observed[0]), cudaMemcpyDeviceToHost, device.stream), "Qwen fixture output copy");
  if (status.ok()) status = runtime(cudaMemcpyAsync(&error_code, device.error, sizeof(error_code), cudaMemcpyDeviceToHost, device.stream), "Qwen fixture error copy");
  if (status.ok()) status = runtime(cudaStreamSynchronize(device.stream), "Qwen fixture synchronize");
  if (!status.ok()) return status;
  if (error_code != 0 || observed != expected)
    return Status::Internal("Qwen deployed INT4 output differs from locked fixture");
  auto output_digest = sha256(std::as_bytes(std::span(observed)));
  if (!output_digest.ok()) return output_digest.status();
  if (output_sha256 != nullptr) *output_sha256 = output_digest->hex();

  if (verify_invalid_launches) {
    const auto expect_launch_rejected = [&](std::uint64_t rejected_m,
                                            std::uint64_t rejected_n,
                                            std::uint64_t rejected_k) -> Status {
      launch_m = rejected_m;
      launch_n = rejected_n;
      launch_k = rejected_k;
      const auto rejected_elements = rejected_m * rejected_n;
      const auto rejected_blocks = static_cast<unsigned int>(
          (rejected_elements + 255U) / 256U);
      auto rejected_status = runtime(
          cudaMemsetAsync(device.error, 0, sizeof(std::uint32_t), device.stream),
          "Qwen fixture rejected launch error clear");
      if (!rejected_status.ok()) return rejected_status;
      const auto rejected_driver = cuLaunchKernel(
          function, rejected_blocks == 0 ? 1U : rejected_blocks, 1, 1,
          256, 1, 1, 0, reinterpret_cast<CUstream>(device.stream), arguments,
          nullptr);
      if (rejected_driver != CUDA_SUCCESS)
        return cuda_driver_status(rejected_driver,
                                  "cuLaunchKernel Qwen rejected fixture");
      std::uint32_t rejected_error = 0;
      rejected_status = runtime(
          cudaMemcpyAsync(&rejected_error, device.error,
                          sizeof(rejected_error), cudaMemcpyDeviceToHost,
                          device.stream),
          "Qwen fixture rejected launch error copy");
      if (rejected_status.ok())
        rejected_status = runtime(
            cudaStreamSynchronize(device.stream),
            "Qwen fixture rejected launch synchronize");
      if (!rejected_status.ok()) return rejected_status;
      if (rejected_error !=
          static_cast<std::uint32_t>(QwenCudaInvariant::kW4A16Launch))
        return Status::Internal(
            "Qwen deployed INT4 invalid launch was not rejected");
      return Status::Ok();
    };
    for (const auto rejected : {
             std::array<std::uint64_t, 3>{0, n, k},
             std::array<std::uint64_t, 3>{m, n + 1, k},
             std::array<std::uint64_t, 3>{m, n, k + 1},
         }) {
      status = expect_launch_rejected(rejected[0], rejected[1], rejected[2]);
      if (!status.ok()) return status;
    }
    launch_m = m;
    launch_n = n;
    launch_k = k;
  }

  if (!verify_reserved_nibble) return Status::Ok();

  const std::uint8_t reserved = static_cast<std::uint8_t>((packed[0] & 0xf0U) | 8U);
  status = runtime(cudaMemcpyAsync(device.packed, &reserved, 1, cudaMemcpyHostToDevice, device.stream), "Qwen fixture reserved nibble copy");
  if (status.ok()) status = runtime(cudaMemsetAsync(device.error, 0, sizeof(std::uint32_t), device.stream), "Qwen fixture second error clear");
  if (!status.ok()) return status;
  driver = cuLaunchKernel(function, blocks, 1, 1, 256, 1, 1, 0,
                          reinterpret_cast<CUstream>(device.stream), arguments,
                          nullptr);
  if (driver != CUDA_SUCCESS) return cuda_driver_status(driver, "cuLaunchKernel Qwen reserved fixture");
  status = runtime(cudaMemcpyAsync(&error_code, device.error, sizeof(error_code), cudaMemcpyDeviceToHost, device.stream), "Qwen fixture reserved error copy");
  if (status.ok()) status = runtime(cudaStreamSynchronize(device.stream), "Qwen fixture reserved synchronize");
  if (!status.ok()) return status;
  if (error_code != static_cast<std::uint32_t>(QwenCudaInvariant::kW4A16ReservedNibble))
    return Status::Internal("Qwen deployed INT4 reserved nibble was not rejected");
  return Status::Ok();
}

Result<std::filesystem::path> deployed_cubin_path() {
  const char* raw_path = std::getenv("PIH_QWEN_CUBIN_PATH");
  if (raw_path == nullptr || *raw_path == '\0')
    return Status::FailedPrecondition(
        "PIH_QWEN_CUBIN_PATH is required for the deployed fixture");
  const std::filesystem::path path(raw_path);
  if (!path.is_absolute() || !std::filesystem::is_regular_file(path) ||
      std::filesystem::file_size(path) > 64U * 1024U * 1024U)
    return Status::InvalidArgument("Qwen deployed CUBIN path is invalid");
  return path;
}

}  // namespace

Status cuda_verify_qwen_int4_gemm_fixture() {
  auto path = deployed_cubin_path();
  if (!path.ok()) return path.status();
  return verify_case(*path, 3, 1024, 1024, true, false, 0, nullptr);
}

Result<QwenInt4ShapeMatrixReceipt> cuda_collect_qwen_int4_gemm_shape_matrix() {
  auto path = deployed_cubin_path();
  if (!path.ok()) return path.status();
  struct Case final {
    std::uint64_t m, n, k, seed;
  };
  constexpr std::array<Case, 5> cases{{
      {1, 2048, 1024, 0x51a7U}, {2, 1024, 1024, 0x62b8U},
      {3, 1024, 2048, 0x73c9U}, {7, 3072, 1024, 0x84daU},
      {17, 1024, 3072, 0x95ebU},
  }};
  QwenInt4ShapeMatrixReceipt receipt{};
  for (std::size_t index = 0; index < cases.size(); ++index) {
    const auto& value = cases[index];
    std::string output_sha256;
    auto status = verify_case(*path, value.m, value.n, value.k, false,
                              index + 1 == cases.size(), value.seed,
                              &output_sha256);
    if (!status.ok()) return status;
    receipt.cases[index] = {
        value.m, value.n, value.k, value.seed, std::move(output_sha256)};
  }
  receipt.negative_launch_invariants = {
      static_cast<std::uint32_t>(QwenCudaInvariant::kW4A16Launch),
      static_cast<std::uint32_t>(QwenCudaInvariant::kW4A16Launch),
      static_cast<std::uint32_t>(QwenCudaInvariant::kW4A16Launch),
  };
  return receipt;
}

Status cuda_verify_qwen_int4_gemm_shape_matrix() {
  auto receipt = cuda_collect_qwen_int4_gemm_shape_matrix();
  return receipt.ok() ? Status::Ok() : receipt.status();
}

Status cuda_verify_qwen_teacher_forced_metric_fixture() {
  auto path = deployed_cubin_path();
  if (!path.ok()) return path.status();
  constexpr std::uint32_t rows = 3;
  constexpr std::uint32_t vocabulary = 151936;
  std::vector<float> logits(static_cast<std::size_t>(rows) * vocabulary);
  for (std::uint32_t row = 0; row < rows; ++row)
    for (std::uint32_t token = 0; token < vocabulary; ++token)
      logits[static_cast<std::size_t>(row) * vocabulary + token] =
          static_cast<float>(static_cast<int>(
              (token * 17U + row * 31U) % 257U) - 128) /
          32.0F;
  logits[17] = 22.0F;
  logits[29] = 22.0F;  // Lowest-token tie must choose token 17.
  logits[static_cast<std::size_t>(vocabulary) + 65537] = 31.0F;
  logits[static_cast<std::size_t>(2) * vocabulary + vocabulary - 1] = 19.0F;
  const std::vector<std::uint32_t> targets{29, 65537, vocabulary - 1};
  auto expected = qwen_teacher_forced_metric_oracle(logits, targets, vocabulary);
  if (!expected.ok()) return expected.status();
  std::vector<std::uint32_t> argmax(rows), nonfinite(rows);
  std::vector<double> nll(rows);
  MetricDeviceMemory device;
  Status status = runtime(cudaMalloc(reinterpret_cast<void**>(&device.logits), logits.size() * sizeof(float)), "Qwen metric fixture cudaMalloc logits");
  if (status.ok()) status = runtime(cudaMalloc(reinterpret_cast<void**>(&device.targets), targets.size() * sizeof(std::uint32_t)), "Qwen metric fixture cudaMalloc targets");
  if (status.ok()) status = runtime(cudaMalloc(reinterpret_cast<void**>(&device.argmax), argmax.size() * sizeof(std::uint32_t)), "Qwen metric fixture cudaMalloc argmax");
  if (status.ok()) status = runtime(cudaMalloc(reinterpret_cast<void**>(&device.nll), nll.size() * sizeof(double)), "Qwen metric fixture cudaMalloc nll");
  if (status.ok()) status = runtime(cudaMalloc(reinterpret_cast<void**>(&device.nonfinite), nonfinite.size() * sizeof(std::uint32_t)), "Qwen metric fixture cudaMalloc nonfinite");
  if (status.ok()) status = runtime(cudaMalloc(reinterpret_cast<void**>(&device.error), sizeof(std::uint32_t)), "Qwen metric fixture cudaMalloc error");
  if (status.ok()) status = runtime(cudaStreamCreateWithFlags(&device.stream, cudaStreamNonBlocking), "Qwen metric fixture stream create");
  if (!status.ok()) return status;
  CUresult driver = cuModuleLoad(&device.module, path->string().c_str());
  if (driver != CUDA_SUCCESS) return cuda_driver_status(driver, "cuModuleLoad Qwen metric fixture");
  CUfunction function = nullptr;
  driver = cuModuleGetFunction(&function, device.module, "pih_qwen_teacher_forced_metric_f32_v1");
  if (driver != CUDA_SUCCESS) return cuda_driver_status(driver, "cuModuleGetFunction Qwen metric fixture");
  status = runtime(cudaMemcpyAsync(device.logits, logits.data(), logits.size() * sizeof(float), cudaMemcpyHostToDevice, device.stream), "Qwen metric fixture logits copy");
  if (status.ok()) status = runtime(cudaMemcpyAsync(device.targets, targets.data(), targets.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice, device.stream), "Qwen metric fixture targets copy");
  if (status.ok()) status = runtime(cudaMemsetAsync(device.error, 0, sizeof(std::uint32_t), device.stream), "Qwen metric fixture error clear");
  if (!status.ok()) return status;
  std::uint32_t launch_rows = rows, launch_vocabulary = vocabulary;
  void* arguments[] = {&device.logits, &device.targets, &device.argmax, &device.nll,
                       &device.nonfinite, &device.error, &launch_rows, &launch_vocabulary};
  driver = cuLaunchKernel(function, rows, 1, 1, 256, 1, 1, 0,
                          reinterpret_cast<CUstream>(device.stream), arguments, nullptr);
  if (driver != CUDA_SUCCESS) return cuda_driver_status(driver, "cuLaunchKernel Qwen metric fixture");
  std::uint32_t error = 0;
  status = runtime(cudaMemcpyAsync(argmax.data(), device.argmax, argmax.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost, device.stream), "Qwen metric fixture argmax copy");
  if (status.ok()) status = runtime(cudaMemcpyAsync(nll.data(), device.nll, nll.size() * sizeof(double), cudaMemcpyDeviceToHost, device.stream), "Qwen metric fixture nll copy");
  if (status.ok()) status = runtime(cudaMemcpyAsync(nonfinite.data(), device.nonfinite, nonfinite.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost, device.stream), "Qwen metric fixture nonfinite copy");
  if (status.ok()) status = runtime(cudaMemcpyAsync(&error, device.error, sizeof(error), cudaMemcpyDeviceToHost, device.stream), "Qwen metric fixture error copy");
  if (status.ok()) status = runtime(cudaStreamSynchronize(device.stream), "Qwen metric fixture synchronize");
  if (!status.ok()) return status;
  if (error != 0) return Status::Internal("Qwen metric fixture reported a launch invariant");
  for (std::size_t row = 0; row < rows; ++row) {
    if (argmax[row] != expected->rows[row].argmax_token ||
        nonfinite[row] != (expected->rows[row].finite ? 0U : 1U) ||
        std::abs(nll[row] - expected->rows[row].target_nll) > 1.0e-9)
      return Status::Internal("Qwen deployed metric output differs from CPU oracle");
  }
  launch_rows = 0;
  status = runtime(cudaMemsetAsync(device.error, 0, sizeof(std::uint32_t), device.stream), "Qwen metric fixture rejected error clear");
  if (!status.ok()) return status;
  driver = cuLaunchKernel(function, 1, 1, 1, 256, 1, 1, 0,
                          reinterpret_cast<CUstream>(device.stream), arguments, nullptr);
  if (driver != CUDA_SUCCESS) return cuda_driver_status(driver, "cuLaunchKernel Qwen rejected metric fixture");
  status = runtime(cudaMemcpyAsync(&error, device.error, sizeof(error), cudaMemcpyDeviceToHost, device.stream), "Qwen metric fixture rejected error copy");
  if (status.ok()) status = runtime(cudaStreamSynchronize(device.stream), "Qwen metric fixture rejected synchronize");
  if (!status.ok()) return status;
  return error == static_cast<std::uint32_t>(QwenCudaInvariant::kTeacherForcedMetricLaunch)
             ? Status::Ok()
             : Status::Internal("Qwen invalid metric launch was not rejected");
}

}  // namespace pih
