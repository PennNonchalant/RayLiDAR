#include "fixture.h"

#include <raylidar/common/io.h>
#include <raylidar/rays/rays.h>
#include <raylidar/rays/build.h>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
namespace fs = std::filesystem;
using raylidar::IoError;
using raylidar::gt::GtPoint;
using raylidar::rays::Bucket;
using raylidar::rays::BucketOf;
using raylidar::rays::BuildRays;
using raylidar::rays::Ray;
using raylidar::rays::ReadRayFrame;
using raylidar::rays::WriteRayFrame;
using raylidar::test::MakeGtPoint;
using raylidar::test::MakeRayTransforms;
using raylidar::test::MakeRaysTempRoot;
using raylidar::test::TranslateRays;

std::vector<std::uint8_t> ReadAllBytes(const fs::path &path)
{
    std::ifstream input(path, std::ios::binary);
    return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(input),
                                     std::istreambuf_iterator<char>());
}

void WriteAllBytes(const fs::path &path, const std::vector<std::uint8_t> &bytes)
{
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char *>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
}
} // namespace

TEST_CASE("bucket edges match D-17 exactly", "[rays]")
{
    CHECK(BucketOf(0.0) == Bucket::B0_20);
    CHECK(BucketOf(19.999) == Bucket::B0_20);
    CHECK(BucketOf(20.0) == Bucket::B20_40);
    CHECK(BucketOf(39.999) == Bucket::B20_40);
    CHECK(BucketOf(40.0) == Bucket::B40_50);
    CHECK(BucketOf(50.0) == Bucket::B40_50);
    CHECK(BucketOf(50.001) == Bucket::B50_60);
    CHECK(BucketOf(60.0) == Bucket::B50_60);

    CHECK_THROWS_AS(BucketOf(-0.001), std::invalid_argument);
    CHECK_THROWS_AS(BucketOf(60.001), std::invalid_argument);
}

TEST_CASE("BuildRays maps through the precomputed transform and keeps a unit direction", "[rays]")
{
    auto transforms = MakeRayTransforms();
    transforms["image_00"] = TranslateRays(1.0, 0.0, 0.0);
    transforms["image_01"] = TranslateRays(2.0, 0.0, 0.0);

    const double expectedRange = std::sqrt(101.0);
    const std::vector<GtPoint> points = {
        MakeGtPoint(0.0F, 0.0F, 10.0F, 0U, 26U, static_cast<float>(expectedRange)),
        MakeGtPoint(0.0F, 0.0F, 10.0F, 1U, 7U, static_cast<float>(std::sqrt(104.0)))};

    const std::vector<Ray> rays = BuildRays(points, 7, transforms);
    REQUIRE(rays.size() == 2U);

    CHECK((rays[0].p_C - Eigen::Vector3f(1.0F, 0.0F, 10.0F)).norm() < 1e-6F);
    CHECK((rays[1].p_C - Eigen::Vector3f(2.0F, 0.0F, 10.0F)).norm() < 1e-6F);
    CHECK(std::abs(rays[0].t_gt - static_cast<float>(expectedRange)) < 1e-5F);
    CHECK(std::abs(rays[1].t_gt - static_cast<float>(std::sqrt(104.0))) < 1e-5F);
    CHECK(std::abs(rays[0].dir_C.norm() - 1.0F) < 1e-6F);
    CHECK(std::abs(rays[1].dir_C.norm() - 1.0F) < 1e-6F);
    CHECK((rays[0].dir_C - rays[0].p_C / rays[0].t_gt).norm() < 1e-6F);
    CHECK(rays[0].camera == 0U);
    CHECK(rays[1].camera == 1U);
    CHECK(rays[0].semanticId == 26U);
    CHECK(rays[1].semanticId == 7U);
    CHECK(rays[0].bucket == static_cast<std::uint8_t>(Bucket::B0_20));
}

TEST_CASE("BuildRays skips camera 255 without subsampling valid points", "[rays]")
{
    const auto transforms = MakeRayTransforms();
    const std::vector<GtPoint> points = {
        MakeGtPoint(0.0F, 0.0F, 10.0F, 0U, 7U, 10.0F),
        MakeGtPoint(5.0F, 0.0F, 10.0F, 255U, 65535U, -1.0F),
        MakeGtPoint(10.0F, 0.0F, 10.0F, 0U, 7U, static_cast<float>(std::sqrt(200.0))),
        MakeGtPoint(15.0F, 0.0F, 10.0F, 1U, 7U, static_cast<float>(std::sqrt(325.0)))};

    const std::vector<Ray> rays = BuildRays(points, 3, transforms);
    REQUIRE(rays.size() == 3U);
    CHECK(std::abs(rays[1].p_L.x() - 10.0F) < 1e-6F);
    CHECK(std::abs(rays[2].p_L.x() - 15.0F) < 1e-6F);
}

TEST_CASE("ray ids are unique within a frame and stable for the same input order", "[rays]")
{
    const auto transforms = MakeRayTransforms();
    const std::vector<GtPoint> points = {
        MakeGtPoint(0.0F, 0.0F, 10.0F, 0U, 7U, 10.0F),
        MakeGtPoint(1.0F, 0.0F, 10.0F, 0U, 7U, static_cast<float>(std::sqrt(101.0))),
        MakeGtPoint(2.0F, 0.0F, 10.0F, 1U, 7U, static_cast<float>(std::sqrt(104.0)))};

    const std::vector<Ray> first = BuildRays(points, 4615, transforms);
    const std::vector<Ray> second = BuildRays(points, 4615, transforms);
    REQUIRE(first.size() == 3U);
    REQUIRE(second.size() == 3U);
    for (std::size_t index = 0; index < first.size(); ++index)
    {
        CHECK(first[index].ray_id == second[index].ray_id);
        CHECK((first[index].ray_id >> 32U) == 4615U);
        CHECK((first[index].ray_id & 0xFFFFFFFFU) == index);
    }
    CHECK(first[0].ray_id != first[1].ray_id);
    CHECK(first[1].ray_id != first[2].ray_id);
}

TEST_CASE("the 64-byte ray file round-trips every field", "[rays]")
{
    const fs::path root = MakeRaysTempRoot("rays_roundtrip");
    auto transforms = MakeRayTransforms();
    transforms["image_00"] = TranslateRays(0.5, 0.0, 0.0);
    const std::vector<Ray> rays =
        BuildRays({MakeGtPoint(0.0F, 0.0F, 40.0F, 0U, 26U, 40.003F),
                   MakeGtPoint(0.0F, 0.0F, 55.0F, 1U, 7U, 55.0F)},
                  9, transforms);
    REQUIRE(rays.size() == 2U);

    const fs::path path = root / "frame_0000000009.bin";
    WriteRayFrame(path, 9, rays);
    const std::vector<std::uint8_t> bytes = ReadAllBytes(path);
    CHECK(bytes.size() == 32U + rays.size() * 64U);
    CHECK(std::string(reinterpret_cast<const char *>(bytes.data()), 4U) == "RLRY");

    int frame = -1;
    const std::vector<Ray> decoded = ReadRayFrame(path, &frame);
    CHECK(frame == 9);
    REQUIRE(decoded.size() == rays.size());
    for (std::size_t index = 0; index < rays.size(); ++index)
    {
        CHECK(decoded[index].ray_id == rays[index].ray_id);
        CHECK((decoded[index].dir_C.array() == rays[index].dir_C.array()).all());
        CHECK(decoded[index].t_gt == rays[index].t_gt);
        CHECK((decoded[index].p_L.array() == rays[index].p_L.array()).all());
        CHECK((decoded[index].p_C.array() == rays[index].p_C.array()).all());
        CHECK(decoded[index].bucket == rays[index].bucket);
        CHECK(decoded[index].occluded == rays[index].occluded);
        CHECK(decoded[index].camera == rays[index].camera);
        CHECK(decoded[index].semanticId == rays[index].semanticId);
    }
}

TEST_CASE("ReadRayFrame rejects truncated and wrong-magic input", "[rays]")
{
    const fs::path root = MakeRaysTempRoot("rays_corrupt");
    const std::vector<Ray> rays =
        BuildRays({MakeGtPoint(0.0F, 0.0F, 10.0F, 0U, 7U, 10.0F)}, 1, MakeRayTransforms());
    const fs::path path = root / "frame_0000000001.bin";
    WriteRayFrame(path, 1, rays);
    const std::vector<std::uint8_t> good = ReadAllBytes(path);

    std::vector<std::uint8_t> truncated = good;
    REQUIRE(!truncated.empty());
    truncated.pop_back();
    WriteAllBytes(path, truncated);
    bool truncatedThrew = false;
    try
    {
        (void)ReadRayFrame(path);
    }
    catch (const IoError &error)
    {
        truncatedThrew = true;
        CHECK(error.exitCode() == 4);
    }
    CHECK(truncatedThrew);

    std::vector<std::uint8_t> badMagic = good;
    REQUIRE(badMagic.size() > 4U);
    badMagic[0] = 'X';
    badMagic[1] = 'X';
    badMagic[2] = 'X';
    badMagic[3] = 'X';
    WriteAllBytes(path, badMagic);
    CHECK_THROWS_AS(ReadRayFrame(path), IoError);
}
