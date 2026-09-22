#include "pih/model/runtime_profile_inherited_fd_manifest.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

TEST(RuntimeProfileInheritedFdManifestTest, FreezesTwelveDistinctRoleFds) {
  const std::array<std::int32_t, 5> authority{3, 4, 5, 6, 7};
  const std::array<std::int32_t, 7> references{8, 9, 10, 11, 12, 13, 14};
  auto first = RuntimeProfileInheritedFdManifest::Create(authority, references);
  auto second = RuntimeProfileInheritedFdManifest::Create(authority, references);
  ASSERT_TRUE(first.ok());
  ASSERT_TRUE(second.ok());
  EXPECT_EQ(first->authority_fds(), authority);
  EXPECT_EQ(first->reference_fds(), references);
  EXPECT_EQ(first->manifest_root(), second->manifest_root());
  EXPECT_FALSE(first->canonical_bytes().empty());

  auto reordered = references;
  std::swap(reordered[0], reordered[1]);
  auto changed = RuntimeProfileInheritedFdManifest::Create(authority, reordered);
  ASSERT_TRUE(changed.ok());
  EXPECT_NE(first->manifest_root(), changed->manifest_root());
}

TEST(RuntimeProfileInheritedFdManifestTest, RejectsReservedAndAliasedFds) {
  std::array<std::int32_t, 5> authority{3, 4, 5, 6, 7};
  std::array<std::int32_t, 7> references{8, 9, 10, 11, 12, 13, 14};
  authority[0] = 2;
  EXPECT_FALSE(RuntimeProfileInheritedFdManifest::Create(authority, references).ok());
  authority[0] = 3;
  references[1] = references[0];
  EXPECT_FALSE(RuntimeProfileInheritedFdManifest::Create(authority, references).ok());
  references[1] = 8;
  references[0] = authority[3];
  EXPECT_FALSE(RuntimeProfileInheritedFdManifest::Create(authority, references).ok());
}

}  // namespace
}  // namespace pih
