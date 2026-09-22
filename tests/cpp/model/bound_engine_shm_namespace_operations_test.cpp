#include "pih/model/bound_engine_shm_namespace_operations.h"

#include <array>
#include <gtest/gtest.h>

namespace pih {
namespace {

class ShmStatBackend final : public EngineShmNamespaceStatBackend {
 public:
  Result<EngineShmNamespaceStat> stat_at(
      std::int32_t directory_descriptor,
      std::string_view basename) override {
    ++calls;
    last_directory_descriptor = directory_descriptor;
    last_basename = basename;
    return result;
  }

  int calls = 0;
  std::int32_t last_directory_descriptor = -1;
  std::string last_basename;
  Result<EngineShmNamespaceStat> result =
      EngineShmNamespaceStat{true, true, 7, 70, 4096};
};

TEST(BoundEngineShmNamespaceOperationsTest, MapsIdentityToSafeRelativeName) {
  ShmStatBackend backend;
  const std::array bindings{
      EngineShmNamespaceHandleBinding{20, 5, "pih-a"},
      EngineShmNamespaceHandleBinding{21, 5, "pih-b"}};
  auto operations = BoundEngineShmNamespaceOperations::Create(
      bindings, backend).value();

  auto observation = operations.observe(21);
  ASSERT_TRUE(observation.ok());
  EXPECT_EQ(observation->object_identity, 21U);
  EXPECT_TRUE(observation->present);
  EXPECT_TRUE(observation->regular_file);
  EXPECT_EQ(backend.last_directory_descriptor, 5);
  EXPECT_EQ(backend.last_basename, "pih-b");
}

TEST(BoundEngineShmNamespaceOperationsTest, PreservesAbsentAndFailure) {
  ShmStatBackend backend;
  const std::array bindings{
      EngineShmNamespaceHandleBinding{20, 5, "pih-a"}};
  auto operations = BoundEngineShmNamespaceOperations::Create(
      bindings, backend).value();
  backend.result = EngineShmNamespaceStat{};
  EXPECT_FALSE(operations.observe(20)->present);
  backend.result = Status::Unavailable("SHM stat unavailable");
  auto unavailable = operations.observe(20);
  ASSERT_FALSE(unavailable.ok());
  EXPECT_EQ(unavailable.status().code(), StatusCode::kUnavailable);
}

TEST(BoundEngineShmNamespaceOperationsTest, RejectsUnknownIdentity) {
  ShmStatBackend backend;
  const std::array bindings{
      EngineShmNamespaceHandleBinding{20, 5, "pih-a"}};
  auto operations = BoundEngineShmNamespaceOperations::Create(
      bindings, backend).value();
  EXPECT_FALSE(operations.observe(21).ok());
  EXPECT_EQ(backend.calls, 0);
}

TEST(BoundEngineShmNamespaceOperationsTest, RejectsUnsafeOrDuplicateBindings) {
  ShmStatBackend backend;
  const std::array<std::string, 6> unsafe_names{
      "", ".", "..", "a/b", "a\\b", std::string("a\0b", 3)};
  for (const auto& name : unsafe_names) {
    const std::array bindings{
        EngineShmNamespaceHandleBinding{20, 5, name}};
    EXPECT_FALSE(BoundEngineShmNamespaceOperations::Create(
                     bindings, backend).ok()) << name;
  }
  std::array bindings{
      EngineShmNamespaceHandleBinding{20, 5, "pih-a"},
      EngineShmNamespaceHandleBinding{21, 5, "pih-b"}};
  bindings[1].object_identity = 20;
  EXPECT_FALSE(BoundEngineShmNamespaceOperations::Create(
                   bindings, backend).ok());
  bindings[1] = {21, 5, "pih-a"};
  EXPECT_FALSE(BoundEngineShmNamespaceOperations::Create(
                   bindings, backend).ok());
}

}  // namespace
}  // namespace pih
