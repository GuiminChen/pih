#include "pih/backend/cuda/deepseek_component_sha256.h"

#include <array>
#include <string_view>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include "pih/core/sha256.h"

namespace pih { namespace {

TEST(DeepSeekComponentSha256Test, MatchesCanonicalSha256OnTargetGpu) {
  if (cudaSetDevice(0) != cudaSuccess) GTEST_SKIP() << "CUDA device unavailable";
  constexpr std::string_view input =
      "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
  const auto expected = sha256(std::as_bytes(std::span(input)));
  ASSERT_TRUE(expected.ok());

  void* device_input = nullptr;
  void* device_output = nullptr;
  void* device_error = nullptr;
  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaMalloc(&device_input, input.size()), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&device_output, sizeof(Sha256Digest)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&device_error, sizeof(std::uint32_t)), cudaSuccess);
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(device_input, input.data(), input.size(),
                            cudaMemcpyHostToDevice, stream), cudaSuccess);
  ASSERT_EQ(cudaMemsetAsync(device_error, 0, sizeof(std::uint32_t), stream),
            cudaSuccess);
  ASSERT_TRUE(launch_deepseek_component_sha256({
      reinterpret_cast<std::uintptr_t>(device_input), input.size(),
      reinterpret_cast<std::uintptr_t>(device_output),
      reinterpret_cast<std::uintptr_t>(device_error),
      reinterpret_cast<std::uintptr_t>(stream)}).ok());
  Sha256Digest actual;
  std::uint32_t error = 0;
  ASSERT_EQ(cudaMemcpyAsync(&actual, device_output, sizeof(actual),
                            cudaMemcpyDeviceToHost, stream), cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(&error, device_error, sizeof(error),
                            cudaMemcpyDeviceToHost, stream), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  EXPECT_EQ(error, 0U);
  EXPECT_EQ(actual, *expected);
  EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
  EXPECT_EQ(cudaFree(device_error), cudaSuccess);
  EXPECT_EQ(cudaFree(device_output), cudaSuccess);
  EXPECT_EQ(cudaFree(device_input), cudaSuccess);
}

} }  // namespace pih
