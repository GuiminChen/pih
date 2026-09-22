#include "pih/model/cgroup_v2_events.h"

#include <gtest/gtest.h>

namespace pih {

TEST(CgroupV2EventsTest, ParsesCanonicalPopulatedWithOtherFields) {
  auto populated = parse_cgroup_v2_populated("populated 1\nfrozen 0\n");
  ASSERT_TRUE(populated.ok());
  EXPECT_TRUE(*populated);
  auto empty = parse_cgroup_v2_populated("populated 0\n");
  ASSERT_TRUE(empty.ok());
  EXPECT_FALSE(*empty);
}

TEST(CgroupV2EventsTest, RejectsMissingDuplicateAndNoncanonicalValues) {
  EXPECT_FALSE(parse_cgroup_v2_populated("frozen 0\n").ok());
  EXPECT_FALSE(parse_cgroup_v2_populated("populated 0\npopulated 0\n").ok());
  EXPECT_FALSE(parse_cgroup_v2_populated("populated 2\n").ok());
  EXPECT_FALSE(parse_cgroup_v2_populated("populated 00\n").ok());
  EXPECT_FALSE(parse_cgroup_v2_populated("populated\n").ok());
  EXPECT_FALSE(parse_cgroup_v2_populated(" populated 0\n").ok());
}

TEST(CgroupV2EventsTest, RejectsEmptyLinesNulAndOversizedInput) {
  EXPECT_FALSE(parse_cgroup_v2_populated("populated 0\n\n").ok());
  EXPECT_FALSE(parse_cgroup_v2_populated(
      std::string_view("populated 0\0frozen 0", 22)).ok());
  EXPECT_FALSE(parse_cgroup_v2_populated(std::string(4097, 'x')).ok());
}

}  // namespace pih
