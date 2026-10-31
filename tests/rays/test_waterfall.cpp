#include <raylidar/common/io.h>
#include <raylidar/rays/waterfall.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace
{
using raylidar::IoError;
using raylidar::rays::AggregateWaterfalls;
using raylidar::rays::BuildFrameWaterfall;
using raylidar::rays::CameraDropClass;
using raylidar::rays::ClassifyCamera255Point;
using raylidar::rays::FrameWaterfall;
using raylidar::rays::ValidateWaterfall;

Eigen::Matrix3d Intrinsics()
{
    Eigen::Matrix3d k = Eigen::Matrix3d::Identity();
    k << 100.0, 0.0, 100.0,
         0.0, 100.0, 100.0,
         0.0, 0.0, 1.0;
    return k;
}

std::map<std::string, Eigen::Matrix4d> Transforms()
{
    return {{"image_00", Eigen::Matrix4d::Identity()},
            {"image_01", Eigen::Matrix4d::Identity()}};
}

std::map<std::string, Eigen::Matrix3d> IntrinsicsMap()
{
    return {{"image_00", Intrinsics()}, {"image_01", Intrinsics()}};
}

std::map<std::string, std::pair<int, int>> Sizes()
{
    return {{"image_00", {200, 200}}, {"image_01", {200, 200}}};
}

nlohmann::json Counts(const std::int64_t nonfinite, const std::int64_t ego,
                      const std::int64_t dynamic, const std::int64_t kept)
{
    return {{"nonfinite", nonfinite},
            {"ego_roi", ego},
            {"dynamic_union", dynamic},
            {"kept_static", kept}};
}
} // namespace

TEST_CASE("waterfall closes exactly and rejects a single added or lost point", "[rays]")
{
    FrameWaterfall waterfall;
    waterfall.frame = 7;
    waterfall.raw = 10;
    waterfall.nonfinite = 1;
    waterfall.ego_roi = 2;
    waterfall.dynamic_union = 3;
    waterfall.no_camera_fov = 1;
    waterfall.range_out_of_domain = 1;
    waterfall.valid = 2;
    std::string reason;
    CHECK(ValidateWaterfall(waterfall, &reason));
    CHECK(reason.empty());

    // Moving a point between exclusive reasons does not change the sum.
    waterfall.nonfinite = 0;
    waterfall.ego_roi = 3;
    CHECK(ValidateWaterfall(waterfall, &reason));
    CHECK(reason.empty());

    waterfall.raw = 11;
    CHECK_FALSE(ValidateWaterfall(waterfall, &reason));
    CHECK_FALSE(reason.empty());
    CHECK(reason.find("difference 1") != std::string::npos);

    waterfall.raw = 10;
    waterfall.valid = 1;
    CHECK_FALSE(ValidateWaterfall(waterfall, nullptr));
}

TEST_CASE("waterfall rejects a negative reason count", "[rays]")
{
    FrameWaterfall waterfall;
    waterfall.frame = 1;
    waterfall.raw = 5;
    waterfall.valid = 5;
    waterfall.ego_roi = -1;
    std::string reason;
    CHECK_FALSE(ValidateWaterfall(waterfall, &reason));
    CHECK(reason.find("negative") != std::string::npos);
}

TEST_CASE("BuildFrameWaterfall enforces both source closures", "[rays]")
{
    const nlohmann::json counts = Counts(0, 1, 1, 2);
    const FrameWaterfall waterfall = BuildFrameWaterfall(0, 4, counts, 2, 2, 0, 0);
    CHECK(waterfall.raw == 4);
    CHECK(waterfall.valid == 2);
    CHECK(ValidateWaterfall(waterfall));

    CHECK_THROWS_AS(BuildFrameWaterfall(0, 5, counts, 2, 2, 0, 0), IoError);
    CHECK_THROWS_AS(BuildFrameWaterfall(0, 4, counts, 2, 1, 0, 0), IoError);

    const nlohmann::json negative = Counts(-1, 1, 1, 2);
    CHECK_THROWS_AS(BuildFrameWaterfall(0, 4, negative, 3, 2, 1, 0), IoError);
}

TEST_CASE("AggregateWaterfalls reports coverage, every reason, and exactness", "[rays]")
{
    FrameWaterfall first;
    first.frame = 0;
    first.raw = 4;
    first.ego_roi = 1;
    first.dynamic_union = 1;
    first.valid = 2;

    FrameWaterfall second = first;
    second.frame = 1;
    second.raw = 5; // deliberately non-closing

    const nlohmann::json aggregate = AggregateWaterfalls({first, second});
    CHECK(aggregate["frames"].size() == 2U);
    CHECK(aggregate["frames"][0]["exact"].get<bool>());
    CHECK_FALSE(aggregate["frames"][1]["exact"].get<bool>());
    CHECK_FALSE(aggregate["totals"]["exact"].get<bool>());
    CHECK_FALSE(aggregate["exact"].get<bool>());
    CHECK(aggregate["totals"]["raw"].get<std::int64_t>() == 9);
    CHECK(aggregate["totals"]["valid"].get<std::int64_t>() == 4);
    CHECK(aggregate["totals"]["coverage"].get<double>() == 4.0 / 9.0);
}

TEST_CASE("camera-255 classification partitions no-FOV from out-of-range points", "[rays]")
{
    const auto transforms = Transforms();
    const auto intrinsics = IntrinsicsMap();
    const auto sizes = Sizes();

    // Behind both cameras -> no camera FOV.
    CHECK(ClassifyCamera255Point(Eigen::Vector3d(0.0, 0.0, -5.0), transforms, intrinsics, sizes) ==
          CameraDropClass::NoCameraFov);

    // Projects outside both images -> no camera FOV.
    CHECK(ClassifyCamera255Point(Eigen::Vector3d(500.0, 0.0, 10.0), transforms, intrinsics,
                                 sizes) == CameraDropClass::NoCameraFov);

    // Inside both images, range 1.0 m is below the 1.5 m domain.
    CHECK(ClassifyCamera255Point(Eigen::Vector3d(0.0, 0.0, 1.0), transforms, intrinsics, sizes) ==
          CameraDropClass::RangeOutOfDomain);

    // Inside both images, range 100 m is beyond the 60 m domain.
    CHECK(ClassifyCamera255Point(Eigen::Vector3d(0.0, 0.0, 100.0), transforms, intrinsics, sizes) ==
          CameraDropClass::RangeOutOfDomain);

    // An eligible point must never be labelled camera 255.
    CHECK_THROWS_AS(
        ClassifyCamera255Point(Eigen::Vector3d(0.0, 0.0, 10.0), transforms, intrinsics, sizes),
        IoError);
}
