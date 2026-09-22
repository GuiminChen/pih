#include "pih/model/deepseek_dense_attention_stage_backend.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace pih { namespace {

class FixedOperations final : public DeepSeekFixedStateBankOperations {
 public:
  Status copy_d2d_async(std::uintptr_t, std::uintptr_t, std::uint64_t,
                        std::uintptr_t) override { return Status::Ok(); }
  Status record_event(std::uintptr_t, std::uintptr_t) override {
    return Status::Ok();
  }
  Result<DeepSeekExpertAsyncStatus> query_event(std::uintptr_t) override {
    return DeepSeekExpertAsyncStatus::kSuccess;
  }
};

class InputOperations final : public DeepSeekAttentionProjectionOperations {
 public:
  explicit InputOperations(std::vector<std::string>& calls) : calls_(&calls) {}
  Status validate_host_error(std::uint32_t*) override { return Status::Ok(); }
  Status zero_u32_async(std::uintptr_t, std::uintptr_t) override {
    calls_->push_back("zero"); return Status::Ok();
  }
  Status quant(DeepSeekFp8ActivationQuantLaunch) override {
    calls_->push_back("input_quant"); return Status::Ok();
  }
  Status gemm(DeepSeekFp8GemmLaunch) override {
    calls_->push_back("input_gemm"); return Status::Ok();
  }
  Status rms(DeepSeekRmsNormLaunch) override { return Status::Ok(); }
  Status head_rms(DeepSeekHeadRmsLaunch) override { return Status::Ok(); }
  Status rotary(DeepSeekRotaryLaunch) override { return Status::Ok(); }
  Status kv_simulate(DeepSeekKvFp8SimulateLaunch) override {
    calls_->push_back("kv_ready"); return Status::Ok();
  }
  Status copy_error_d2h_async(std::uint32_t*, std::uintptr_t,
                              std::uintptr_t) override { return Status::Ok(); }
 private:
  std::vector<std::string>* calls_;
};

class OutputOperations final
    : public DeepSeekAttentionOutputProjectionOperations {
 public:
  explicit OutputOperations(std::vector<std::string>& calls) : calls_(&calls) {}
  Status validate_host_error(std::uint32_t*) override { return Status::Ok(); }
  Status zero_u32_async(std::uintptr_t, std::uintptr_t) override {
    calls_->push_back("unexpected_zero"); return Status::Ok();
  }
  Status rotary(DeepSeekRotaryLaunch) override {
    calls_->push_back("inverse_rope"); return Status::Ok();
  }
  Status grouped_gemm(DeepSeekGroupedFp8GemmLaunch) override {
    calls_->push_back("wo_a"); return Status::Ok();
  }
  Status quant(DeepSeekFp8ActivationQuantLaunch) override { return Status::Ok(); }
  Status gemm(DeepSeekFp8GemmLaunch) override {
    calls_->push_back("wo_b"); return Status::Ok();
  }
  Status copy_error_d2h_async(std::uint32_t*, std::uintptr_t,
                              std::uintptr_t) override { return Status::Ok(); }
 private:
  std::vector<std::string>* calls_;
};

class Inner final : public DeepSeekStageOperatorBackend {
 public:
  explicit Inner(std::vector<std::string>& calls) : calls_(&calls) {}
  Status launch(const DeepSeekStageOperatorCommand& command,
                const DeepSeekPipelinePlanDescriptor&) override {
    kind = command.kind; calls_->push_back("sparse_launch"); return Status::Ok();
  }
  Result<DeepSeekStageComputeStatus> poll() override {
    calls_->push_back("sparse_poll");
    return ++polls == 1 ? DeepSeekStageComputeStatus::kInProgress
                        : DeepSeekStageComputeStatus::kSuccess;
  }
  DeepSeekStageOperatorKind kind = DeepSeekStageOperatorKind::kAttention;
  std::uint32_t polls = 0;
 private:
  std::vector<std::string>* calls_;
};

DeepSeekAttentionProjectionSubmission input_submission(
    std::uintptr_t error, std::uintptr_t stream) {
  return {{1, 2, 3, error, stream, 2, 4096},
          {2, 3, 10, 11, 12, error, stream, 2, 1024, 4096},
          {12, 13, 14, error, stream, 2, 1024, 1.0e-6F},
          {14, 15, 16, error, stream, 2, 1024},
          {15, 16, 17, 18, 19, error, stream, 2, 32768, 1024},
          {19, 19, error, stream, 2, 64, 512, 1.0e-6F},
          {19, 20, error, stream, 2, 64, 512, 64, false, 21, 104},
          {2, 3, 21, 22, 23, error, stream, 2, 512, 4096},
          {23, 24, 23, error, stream, 2, 512, 1.0e-6F},
          {23, 20, error, stream, 2, 1, 512, 64, false, 21, 104},
          {23, error, stream, 2, 512, 448, 64}};
}

DeepSeekAttentionOutputProjectionSubmission output_submission(
    std::uintptr_t error, std::uintptr_t stream) {
  DeepSeekAttentionOutputProjectionSubmission value;
  value.inverse_rope = {31, 20, error, stream, 2, 64, 512, 64, true, 21, 104};
  value.wo_a = {31, 32, 33, 34, 35, 36, error, stream,
                2, 8, 1024, 4096};
  value.quant = {36, 37, 38, error, stream, 2, 8192};
  value.wo_b = {37, 38, 39, 40, 41, error, stream, 2, 4096, 8192};
  return value;
}

struct SequenceFixture final {
  SequenceFixture(std::uint32_t sequence, std::uintptr_t error,
                  std::vector<std::string>& calls)
      : input_operations(calls), output_operations(calls),
        transaction(DeepSeekAttentionSequenceTransaction::Create(
            sequence, 0, 0, banks, ratio4, ratio128).value()),
        input_coordinator(DeepSeekAttentionProjectionCoordinator::Create(
            input_operations, &host_error).value()),
        output_coordinator(
            DeepSeekAttentionOutputProjectionCoordinator::Create(
                output_operations, &host_error).value()) {
    work.input_coordinator = &input_coordinator;
    work.output_coordinator = &output_coordinator;
    work.transaction = &transaction;
    work.input = input_submission(error, 11);
    work.output = output_submission(error, 11);
    work.sparse_query_bf16 = 19;
    work.sparse_kv_bf16 = 23;
    work.sparse_output_bf16 = 31;
  }
  FixedOperations fixed;
  DeepSeekFixedStateBanks banks = DeepSeekFixedStateBanks::Create(
      {0x1000, 256}, {0x2000, 256}, 91, fixed).value();
  DeepSeekRatio4PagePool ratio4 = DeepSeekRatio4PagePool::Create(1).value();
  DeepSeekRatio128PagePool ratio128 = DeepSeekRatio128PagePool::Create(1).value();
  InputOperations input_operations;
  OutputOperations output_operations;
  DeepSeekAttentionSequenceTransaction transaction;
  std::uint32_t host_error = 0;
  DeepSeekAttentionProjectionCoordinator input_coordinator;
  DeepSeekAttentionOutputProjectionCoordinator output_coordinator;
  DeepSeekDenseAttentionStageSequenceWork work;
};

class Provider final : public DeepSeekDenseAttentionStageWorkProvider {
 public:
  Result<std::span<const DeepSeekDenseAttentionStageSequenceWork>> resolve(
      const DeepSeekStageOperatorCommand&,
      const DeepSeekPipelinePlanDescriptor&) override { return work; }
  std::span<const DeepSeekDenseAttentionStageSequenceWork> work;
};

DeepSeekPipelinePlanDescriptor plan(std::uint32_t sequences = 1) {
  return {3, 7, DeepSeekPlanPhase::kDecode, sequences, sequences};
}

TEST(DeepSeekDenseAttentionStageBackendTest,
     QueuesProjectionSparseAttentionAndOutputProjectionInOrder) {
  std::vector<std::string> calls;
  SequenceFixture fixture(1, 90, calls);
  ASSERT_TRUE(fixture.transaction.begin(11).ok());
  Provider provider; provider.work = {&fixture.work, 1};
  Inner inner(calls);
  auto backend = DeepSeekDenseAttentionStageOperatorBackend::Create(
      inner, provider).value();

  ASSERT_TRUE(backend.launch({DeepSeekStageOperatorKind::kAttention, 4},
                             plan()).ok());
  EXPECT_EQ(calls.back(), "sparse_launch");
  EXPECT_EQ(backend.poll().value(), DeepSeekStageComputeStatus::kInProgress);
  EXPECT_EQ(backend.poll().value(), DeepSeekStageComputeStatus::kSuccess);
  EXPECT_LT(std::find(calls.begin(), calls.end(), "kv_ready"),
            std::find(calls.begin(), calls.end(), "sparse_launch"));
  EXPECT_LT(std::find(calls.begin(), calls.end(), "sparse_poll"),
            std::find(calls.begin(), calls.end(), "inverse_rope"));
  EXPECT_EQ(calls.back(), "wo_b");
  EXPECT_EQ(std::count(calls.begin(), calls.end(), "zero"), 1);
  EXPECT_EQ(std::count(calls.begin(), calls.end(), "unexpected_zero"), 0);
}

TEST(DeepSeekDenseAttentionStageBackendTest,
     RejectsBrokenSparsePointerFlowBeforeAnySubmission) {
  std::vector<std::string> calls;
  SequenceFixture fixture(1, 90, calls);
  ASSERT_TRUE(fixture.transaction.begin(11).ok());
  fixture.work.sparse_query_bf16 = 999;
  Provider provider; provider.work = {&fixture.work, 1};
  Inner inner(calls);
  auto backend = DeepSeekDenseAttentionStageOperatorBackend::Create(
      inner, provider).value();
  EXPECT_FALSE(backend.launch({DeepSeekStageOperatorKind::kAttention, 4},
                              plan()).ok());
  EXPECT_TRUE(calls.empty());
}

TEST(DeepSeekDenseAttentionStageBackendTest,
     TransparentlyPassesNonAttentionOperators) {
  std::vector<std::string> calls;
  Provider provider;
  Inner inner(calls);
  auto backend = DeepSeekDenseAttentionStageOperatorBackend::Create(
      inner, provider).value();
  ASSERT_TRUE(backend.launch({DeepSeekStageOperatorKind::kMoe, 4}, plan()).ok());
  EXPECT_EQ(inner.kind, DeepSeekStageOperatorKind::kMoe);
  EXPECT_EQ(backend.poll().value(), DeepSeekStageComputeStatus::kInProgress);
  EXPECT_EQ(backend.poll().value(), DeepSeekStageComputeStatus::kSuccess);
}

}}  // namespace pih::<anonymous>
