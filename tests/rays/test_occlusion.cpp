#include "fixture.h"

#include <raylidar/rays/build.h>
#include <raylidar/rays/occlusion.h>
#include <raylidar/rays/rays.h>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace
{
using raylidar::gt::GtPoint;
using raylidar::rays::BuildRays;
using raylidar::rays::ComputeOcclusionFlags;
using raylidar::rays::OcclusionOptions;
using raylidar::rays::Ray;
using raylidar::test::MakeGtPoint;
using raylidar::test::MakeOcclusionIntrinsics;
using raylidar::test::MakeRayTransforms;

constexpr int kWidth = 200;
constexpr int kHeight = 200;

std::vector<std::uint8_t> Flags(const std::vector<GtPoint> &points, const std::vector<Ray> &rays, const int camera,
                                const OcclusionOptions &options = {})
{
    return ComputeOcclusionFlags(points, rays, camera, Eigen::Matrix4d::Identity(), MakeOcclusionIntrinsics(), kWidth,
                                 kHeight, options);
}

GtPoint Point(const float x, const float y, const float z, const std::uint8_t camera)
{
    const float range = std::sqrt(x * x + y * y + z * z);
    return MakeGtPoint(x, y, z, camera, 7U, range);
}
} // namespace

TEST_CASE("a plane with no occluders yields zero flags", "[rays]")
{
    const std::vector<GtPoint> points = {Point(0.0F, 0.0F, 10.0F, 0U), Point(5.0F, 0.0F, 10.0F, 0U),
                                         Point(-5.0F, 0.0F, 10.0F, 0U)};
    const std::vector<Ray> rays = BuildRays(points, 0, MakeRayTransforms());
    REQUIRE(rays.size() == 3U);

    const std::vector<std::uint8_t> flags = Flags(points, rays, 0);
    REQUIRE(flags.size() == rays.size());
    for (const std::uint8_t flag : flags)
    {
        CHECK(flag == 0U);
    }
}

TEST_CASE("a point assigned to the other camera still occludes (all-visible z-buffer)", "[rays]")
{
    const std::vector<GtPoint> points = {Point(0.0F, 0.0F, 10.0F, 0U), Point(0.01F, 0.0F, 5.0F, 1U)};
    const std::vector<Ray> rays = BuildRays(points, 0, MakeRayTransforms());
    REQUIRE(rays.size() == 2U);
    REQUIRE(rays[0].camera == 0U);
    REQUIRE(rays[1].camera == 1U);

    const std::vector<std::uint8_t> camera0 = Flags(points, rays, 0);
    REQUIRE(camera0.size() == 2U);
    CHECK(camera0[0] == 1U);
    CHECK(camera0[1] == 0U);

    const std::vector<std::uint8_t> camera1 = Flags(points, rays, 1);
    REQUIRE(camera1.size() == 2U);
    CHECK(camera1[0] == 0U);
    CHECK(camera1[1] == 0U);
}

TEST_CASE("an occluder closer by less than the threshold does not flag", "[rays]")
{
    const std::vector<GtPoint> points = {Point(0.0F, 0.0F, 10.0F, 0U), Point(0.01F, 0.0F, 9.95F, 1U)};
    const std::vector<Ray> rays = BuildRays(points, 0, MakeRayTransforms());
    REQUIRE(rays.size() == 2U);

    const std::vector<std::uint8_t> flags = Flags(points, rays, 0);
    REQUIRE(flags.size() == 2U);
    CHECK(flags[0] == 0U);
}

TEST_CASE("an occluder outside the pixel neighbourhood does not flag", "[rays]")
{
    const std::vector<GtPoint> points = {Point(0.0F, 0.0F, 10.0F, 0U), Point(0.5F, 0.0F, 5.0F, 1U)};
    const std::vector<Ray> rays = BuildRays(points, 0, MakeRayTransforms());
    REQUIRE(rays.size() == 2U);

    const std::vector<std::uint8_t> flags = Flags(points, rays, 0);
    CHECK(flags[0] == 0U);
}

TEST_CASE("a point behind the camera is never considered", "[rays]")
{
    const std::vector<GtPoint> points = {Point(0.0F, 0.0F, 10.0F, 0U),
                                         MakeGtPoint(0.0F, 0.0F, -5.0F, 255U, 65535U, -1.0F)};
    const std::vector<Ray> rays = BuildRays(points, 0, MakeRayTransforms());
    REQUIRE(rays.size() == 1U);

    const std::vector<std::uint8_t> flags = Flags(points, rays, 0);
    REQUIRE(flags.size() == 1U);
    CHECK(flags[0] == 0U);
}

TEST_CASE("occlusion rejects an invalid camera id", "[rays]")
{
    const std::vector<GtPoint> points = {Point(0.0F, 0.0F, 10.0F, 0U)};
    const std::vector<Ray> rays = BuildRays(points, 0, MakeRayTransforms());
    CHECK_THROWS_AS(Flags(points, rays, 2), std::invalid_argument);
}
