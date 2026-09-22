#include "pih/model/bound_engine_artifact_descriptor_operations.h"

#include <array>
#include <gtest/gtest.h>

namespace pih {
namespace {

class StatBackend final : public EngineArtifactDescriptorStatBackend {
 public:
  Result<EngineArtifactDescriptorStat> stat(std::int32_t descriptor) override {
    ++calls;
    last_descriptor = descriptor;
    return result;
  }

  int calls = 0;
  std::int32_t last_descriptor = -1;
  Result<EngineArtifactDescriptorStat> result =
      EngineArtifactDescriptorStat{true, true, 4, 40, 4096};
};

TEST(BoundEngineArtifactDescriptorOperationsTest, MapsIdentityToDescriptor) {
  StatBackend backend;
  const std::array bindings{
      EngineArtifactDescriptorHandleBinding{10, 7},
      EngineArtifactDescriptorHandleBinding{11, 8}};
  auto operations = BoundEngineArtifactDescriptorOperations::Create(
      bindings, backend).value();

  auto observation = operations.observe(11);
  ASSERT_TRUE(observation.ok());
  EXPECT_EQ(observation->descriptor_identity, 11U);
  EXPECT_TRUE(observation->open);
  EXPECT_TRUE(observation->regular_file);
  EXPECT_EQ(observation->device_identity, 4U);
  EXPECT_EQ(observation->inode_identity, 40U);
  EXPECT_EQ(observation->size_bytes, 4096U);
  EXPECT_EQ(backend.last_descriptor, 8);
}

TEST(BoundEngineArtifactDescriptorOperationsTest, PreservesClosedAndFailure) {
  StatBackend backend;
  const std::array bindings{EngineArtifactDescriptorHandleBinding{10, 7}};
  auto operations = BoundEngineArtifactDescriptorOperations::Create(
      bindings, backend).value();
  backend.result = EngineArtifactDescriptorStat{false, false, 0, 0, 0};
  auto closed = operations.observe(10);
  ASSERT_TRUE(closed.ok());
  EXPECT_FALSE(closed->open);

  backend.result = Status::Unavailable("descriptor stat unavailable");
  auto unavailable = operations.observe(10);
  ASSERT_FALSE(unavailable.ok());
  EXPECT_EQ(unavailable.status().code(), StatusCode::kUnavailable);
}

TEST(BoundEngineArtifactDescriptorOperationsTest, RejectsUnknownIdentity) {
  StatBackend backend;
  const std::array bindings{EngineArtifactDescriptorHandleBinding{10, 7}};
  auto operations = BoundEngineArtifactDescriptorOperations::Create(
      bindings, backend).value();

  auto observation = operations.observe(12);
  ASSERT_FALSE(observation.ok());
  EXPECT_EQ(observation.status().code(), StatusCode::kFailedPrecondition);
  EXPECT_EQ(backend.calls, 0);
}

TEST(BoundEngineArtifactDescriptorOperationsTest, RejectsInvalidBindings) {
  StatBackend backend;
  std::array bindings{EngineArtifactDescriptorHandleBinding{10, 7},
                      EngineArtifactDescriptorHandleBinding{11, 8}};
  bindings[1].descriptor_identity = 10;
  EXPECT_FALSE(BoundEngineArtifactDescriptorOperations::Create(
                   bindings, backend).ok());
  bindings = {EngineArtifactDescriptorHandleBinding{10, 7},
              EngineArtifactDescriptorHandleBinding{11, 7}};
  EXPECT_FALSE(BoundEngineArtifactDescriptorOperations::Create(
                   bindings, backend).ok());
  bindings[1].descriptor = -1;
  EXPECT_FALSE(BoundEngineArtifactDescriptorOperations::Create(
                   bindings, backend).ok());
}

}  // namespace
}  // namespace pih
