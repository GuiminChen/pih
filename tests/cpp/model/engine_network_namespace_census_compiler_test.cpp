#include "pih/model/engine_network_namespace_census_compiler.h"

#include <limits>
#include <gtest/gtest.h>

namespace pih {
namespace {

Sha256Digest network_compiler_digest(std::uint8_t value) {
  Sha256Digest digest{};
  digest.bytes.fill(static_cast<std::byte>(value));
  return digest;
}

class NetworkBackend final : public EngineNetworkNamespaceCensusBackend {
 public:
  Result<EngineNetworkNamespaceRawSnapshot> capture(
      std::uint64_t namespace_identity) override {
    ++calls;
    last_identity = namespace_identity;
    return result;
  }

  int calls = 0;
  std::uint64_t last_identity = 0;
  Result<EngineNetworkNamespaceRawSnapshot> result =
      EngineNetworkNamespaceRawSnapshot{
          50, true, false,
          {{true, network_compiler_digest(1), network_compiler_digest(2),
            EngineNetworkSocketLifecycle::kActive, true, 100},
           {true, network_compiler_digest(1), network_compiler_digest(2),
            EngineNetworkSocketLifecycle::kActive, true, 120},
           {true, network_compiler_digest(1), network_compiler_digest(3),
            EngineNetworkSocketLifecycle::kTimeWait, true, 80}}};
};

TEST(EngineNetworkNamespaceCensusCompilerTest, AggregatesOwnerLifecycleRows) {
  NetworkBackend backend;
  auto compiler = EngineNetworkNamespaceCensusCompiler::Create(50, backend)
                      .value();
  auto snapshot = compiler.capture(50);
  ASSERT_TRUE(snapshot.ok());
  ASSERT_EQ(snapshot->owners.size(), 2U);
  EXPECT_EQ(snapshot->owners[0].object_count, 2U);
  EXPECT_EQ(snapshot->owners[0].backing_bytes, 220U);
  EXPECT_EQ(snapshot->owners[0].state,
            EngineOwnedResourceLifecycleState::kActive);
  EXPECT_EQ(snapshot->owners[1].object_count, 1U);
  EXPECT_EQ(snapshot->owners[1].state,
            EngineOwnedResourceLifecycleState::kRetired);
}

TEST(EngineNetworkNamespaceCensusCompilerTest, UnknownOwnershipIsUnavailableAndRecovers) {
  NetworkBackend backend;
  backend.result->rows[0].ownership_complete = false;
  auto compiler = EngineNetworkNamespaceCensusCompiler::Create(50, backend)
                      .value();
  auto unavailable = compiler.capture(50);
  ASSERT_FALSE(unavailable.ok());
  EXPECT_EQ(unavailable.status().code(), StatusCode::kUnavailable);
  backend.result->rows[0].ownership_complete = true;
  EXPECT_TRUE(compiler.capture(50).ok());
}

TEST(EngineNetworkNamespaceCensusCompilerTest, UnknownKernelBackingIsUnavailable) {
  NetworkBackend backend;
  backend.result->rows[0].kernel_backing_complete = false;
  auto compiler = EngineNetworkNamespaceCensusCompiler::Create(50, backend)
                      .value();
  auto unavailable = compiler.capture(50);
  ASSERT_FALSE(unavailable.ok());
  EXPECT_EQ(unavailable.status().code(), StatusCode::kUnavailable);
}

TEST(EngineNetworkNamespaceCensusCompilerTest, PreservesDestroyedNamespaceProof) {
  NetworkBackend backend;
  backend.result = EngineNetworkNamespaceRawSnapshot{50, false, true, {}};
  auto compiler = EngineNetworkNamespaceCensusCompiler::Create(50, backend)
                      .value();
  auto snapshot = compiler.capture(50);
  ASSERT_TRUE(snapshot.ok());
  EXPECT_TRUE(snapshot->namespace_destroyed);
  EXPECT_TRUE(snapshot->owners.empty());
}

TEST(EngineNetworkNamespaceCensusCompilerTest, RejectsMalformedRawSnapshot) {
  for (int mutation = 0; mutation < 5; ++mutation) {
    NetworkBackend backend;
    auto snapshot = *backend.result;
    if (mutation == 0) ++snapshot.namespace_identity;
    if (mutation == 1) snapshot.namespace_destroyed = true;
    if (mutation == 2) snapshot.rows[0].owner_identity = {};
    if (mutation == 3)
      snapshot.rows[0].lifecycle =
          static_cast<EngineNetworkSocketLifecycle>(2);
    if (mutation == 4) {
      snapshot.rows[0].kernel_backing_bytes =
          std::numeric_limits<std::uint64_t>::max();
      snapshot.rows[1].kernel_backing_bytes = 1;
    }
    backend.result = snapshot;
    auto compiler = EngineNetworkNamespaceCensusCompiler::Create(50, backend)
                        .value();
    EXPECT_FALSE(compiler.capture(50).ok()) << mutation;
  }
}

TEST(EngineNetworkNamespaceCensusCompilerTest, RejectsWrongRequestedIdentity) {
  NetworkBackend backend;
  auto compiler = EngineNetworkNamespaceCensusCompiler::Create(50, backend)
                      .value();
  EXPECT_FALSE(compiler.capture(51).ok());
  EXPECT_EQ(backend.calls, 0);
  EXPECT_FALSE(EngineNetworkNamespaceCensusCompiler::Create(0, backend).ok());
}

}  // namespace
}  // namespace pih
