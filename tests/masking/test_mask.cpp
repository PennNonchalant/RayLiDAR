#include "fixture.h"

#include <raylidar/calib/kitti360.h>
#include <raylidar/common/mask_bits.h>
#include <raylidar/lidar/velodyne.h>
#include <raylidar/masking/bbox.h>
#include <raylidar/masking/mask.h>
#include <raylidar/masking/semantics.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace
{
namespace fs = std::filesystem;
using Catch::Approx;
using raylidar::calib::Kitti360Calib;
using raylidar::lidar::PointXYZI;
using raylidar::masking::BuildMask;
using raylidar::masking::DynamicBBox;
using raylidar::masking::IsDynamic;
using raylidar::masking::IsEgo;
using raylidar::masking::IsKeptStatic;
using raylidar::masking::kDynamic2D;
using raylidar::masking::kDynamicBBox;
using raylidar::masking::kEgoRoi;
using raylidar::masking::kInvalidNonFinite;
using raylidar::masking::kKeptStatic;
using raylidar::masking::LoadDynamicBBoxes;
using raylidar::masking::MaskOptions;
using raylidar::masking::MaskResult;
using raylidar::masking::SemanticImage;
using raylidar::test::BBoxSpec;
using raylidar::test::MakeBBoxXml;
using raylidar::test::MakeMaskingTempRoot;
using raylidar::test::TranslatedScaled;
using raylidar::test::WriteTextFile;

constexpr int kWidth = 8;
constexpr int kHeight = 8;
constexpr int kFrame = 7;

Kitti360Calib TestCalib()
{
    Kitti360Calib calib;
    Eigen::Matrix3d intrinsics = Eigen::Matrix3d::Zero();
    intrinsics << 2.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 0.0, 1.0;
    calib.K_rect["image_00"] = intrinsics;
    calib.K_rect["image_01"] = intrinsics;
    return calib;
}

SemanticImage LabelImage(const std::map<std::pair<int, int>, std::uint16_t> &overrides,
                         const std::uint16_t background = 7U)
{
    SemanticImage image;
    image.width = kWidth;
    image.height = kHeight;
    image.labels.assign(static_cast<std::size_t>(kWidth) * static_cast<std::size_t>(kHeight), background);
    for (const auto &[pixel, label] : overrides)
    {
        image.labels[static_cast<std::size_t>(pixel.second) * kWidth + static_cast<std::size_t>(pixel.first)] = label;
    }
    return image;
}

std::map<std::string, raylidar::masking::SemanticImage> Semantics()
{
    return {{"image_00", LabelImage({{std::make_pair(2, 2), 26U}, {std::make_pair(2, 3), 26U}})},
            {"image_01", LabelImage({})}};
}

std::map<std::string, std::pair<int, int>> ImageSizes()
{
    return {{"image_00", {kWidth, kHeight}}, {"image_01", {kWidth, kHeight}}};
}

std::map<std::string, Eigen::Matrix4d> CameraTransforms()
{
    return {{"image_00", Eigen::Matrix4d::Identity()}, {"image_01", Eigen::Matrix4d::Identity()}};
}

PointXYZI Point(const double x, const double y, const double z)
{
    return PointXYZI{static_cast<float>(x), static_cast<float>(y), static_cast<float>(z), 1.0F};
}

std::vector<DynamicBBox> ThreeBoxes()
{
    BBoxSpec a;
    a.label = "car";
    a.timestamp = kFrame;
    a.startFrame = 0;
    a.endFrame = 100;
    a.T_W_B = TranslatedScaled(0.0, 0.0, 4.0, 1.0);
    BBoxSpec b = a;
    b.T_W_B = TranslatedScaled(2.0, 0.0, 4.0, 1.0);
    BBoxSpec d = a;
    d.T_W_B = TranslatedScaled(-2.0, 0.0, 4.0, 1.0);

    const fs::path root = MakeMaskingTempRoot("mask_boxes");
    const fs::path path = root / "boxes.xml";
    WriteTextFile(path, MakeBBoxXml({a, b, d}));
    return LoadDynamicBBoxes(path);
}
} // namespace

TEST_CASE("BuildMask records independent dynamic sources and their overlap", "[masking]")
{
    const std::vector<PointXYZI> raw = {
        Point(0.0, 0.0, 4.0),                                      // both sources
        Point(2.0, 0.0, 4.0),                                      // bbox only
        Point(0.0, 2.0, 4.0),                                      // 2D only
        Point(-2.0, 0.0, 4.0),                                     // bbox only
        Point(6.0, 6.0, 4.0),                                      // kept static
        Point(0.0, 0.0, 0.0),                                      // ego ROI
        Point(std::numeric_limits<double>::quiet_NaN(), 0.0, 4.0), // non-finite
    };
    std::vector<Eigen::Vector3d> uncurled;
    uncurled.reserve(raw.size());
    for (const PointXYZI &p : raw)
    {
        uncurled.emplace_back(p.x, p.y, p.z);
    }

    MaskOptions options;
    const MaskResult result = BuildMask(raw, uncurled, kFrame, TestCalib(), Eigen::Matrix4d::Identity(), ThreeBoxes(),
                                        Semantics(), ImageSizes(), CameraTransforms(), options);

    REQUIRE(result.bits.size() == raw.size());
    CHECK((result.bits[0] & kDynamic2D) != 0U);
    CHECK((result.bits[0] & kDynamicBBox) != 0U);
    CHECK((result.bits[1] & kDynamicBBox) != 0U);
    CHECK((result.bits[1] & kDynamic2D) == 0U);
    CHECK((result.bits[2] & kDynamic2D) != 0U);
    CHECK((result.bits[2] & kDynamicBBox) == 0U);
    CHECK((result.bits[3] & kDynamicBBox) != 0U);
    CHECK(IsKeptStatic(result.bits[4]));
    CHECK(IsEgo(result.bits[5]));
    CHECK(result.bits[6] == static_cast<std::uint8_t>(kInvalidNonFinite));

    const auto &counts = result.counts;
    CHECK(counts["nonfinite"].get<std::size_t>() == 1U);
    CHECK(counts["ego_roi"].get<std::size_t>() == 1U);
    CHECK(counts["dynamic_2d"].get<std::size_t>() == 2U);
    CHECK(counts["dynamic_bbox"].get<std::size_t>() == 3U);
    CHECK(counts["dynamic_union"].get<std::size_t>() == 4U);
    CHECK(counts["kept_static"].get<std::size_t>() == 1U);
    CHECK(counts["overlap_2d_only"].get<std::size_t>() == 1U);
    CHECK(counts["overlap_bbox_only"].get<std::size_t>() == 2U);
    CHECK(counts["overlap_both"].get<std::size_t>() == 1U);
    CHECK(counts["overlap_2d_only"].get<std::size_t>() + counts["overlap_bbox_only"].get<std::size_t>() +
              counts["overlap_both"].get<std::size_t>() ==
          counts["dynamic_union"].get<std::size_t>());

    REQUIRE(result.image_00_pixels.size() == static_cast<std::size_t>(kWidth) * kHeight);
    CHECK(result.image_00_pixels[2U * kWidth + 2U] == 1U);
    CHECK(result.image_00_pixels[2U * kWidth + 3U] == 1U);
    CHECK(result.image_00_pixels[7U * kWidth + 7U] == 0U);

    CHECK(result.residual["inside_bbox"].get<std::size_t>() == 0U);
    CHECK(result.residual["within_radius"].get<std::size_t>() == 0U);
    CHECK(result.residual["fraction"].get<double>() == Approx(0.0));
}

TEST_CASE("BuildMask labels dynamic semantics beyond 60 m", "[masking]")
{
    const std::vector<PointXYZI> raw = {Point(0.0, 0.0, 100.0)};
    const std::vector<Eigen::Vector3d> uncurled = {Eigen::Vector3d(0.0, 0.0, 100.0)};

    const MaskResult result = BuildMask(raw, uncurled, kFrame, TestCalib(), Eigen::Matrix4d::Identity(), {},
                                        Semantics(), ImageSizes(), CameraTransforms(), MaskOptions{});

    CHECK((result.bits[0] & kDynamic2D) != 0U);
    CHECK(IsDynamic(result.bits[0]));
    CHECK_FALSE(IsKeptStatic(result.bits[0]));
}

TEST_CASE("BuildMask ego ROI is an exact box, never a radius", "[masking]")
{
    struct Case
    {
        double x;
        double y;
        double z;
        bool ego;
    };
    const std::vector<Case> cases = {
        {2.5, 0.0, 0.0, false}, {2.4999, 0.0, 0.0, true}, {0.0, 1.2, 0.0, false},  {0.0, 1.1999, 0.0, true},
        {0.0, 0.0, 0.3, false}, {0.0, 0.0, 0.2999, true}, {0.0, 0.0, -2.0, false}, {0.0, 0.0, -1.9999, true},
    };

    std::vector<PointXYZI> raw;
    std::vector<Eigen::Vector3d> uncurled;
    for (const Case &entry : cases)
    {
        raw.push_back(Point(entry.x, entry.y, entry.z));
        uncurled.emplace_back(entry.x, entry.y, entry.z);
    }

    const MaskResult result = BuildMask(raw, uncurled, kFrame, TestCalib(), Eigen::Matrix4d::Identity(), {},
                                        Semantics(), ImageSizes(), CameraTransforms(), MaskOptions{});

    for (std::size_t index = 0; index < cases.size(); ++index)
    {
        CHECK(IsEgo(result.bits[index]) == cases[index].ego);
    }
}

TEST_CASE("BuildMask reports residual dynamic proximity among kept-static points", "[masking]")
{
    const std::vector<PointXYZI> raw = {Point(0.0, 0.0, 4.0), Point(-1.5, 0.0, 4.0)};
    const std::vector<Eigen::Vector3d> uncurled = {Eigen::Vector3d(0.0, 0.0, 4.0), Eigen::Vector3d(-1.5, 0.0, 4.0)};
    auto semantics = Semantics();
    semantics["image_00"] = LabelImage({{std::make_pair(2, 2), 26U}, {std::make_pair(1, 2), 7U}});

    MaskOptions options;
    options.residualDynamicRadiusM = 2.0;
    const MaskResult result = BuildMask(raw, uncurled, kFrame, TestCalib(), Eigen::Matrix4d::Identity(), {}, semantics,
                                        ImageSizes(), CameraTransforms(), options);

    CHECK(IsKeptStatic(result.bits[1]));
    CHECK(result.residual["within_radius"].get<std::size_t>() == 1U);
    CHECK(result.residual["inside_bbox"].get<std::size_t>() == 0U);
    CHECK(result.residual["fraction"].get<double>() == Approx(1.0));
}

TEST_CASE("Mask query helpers read only the frozen bits", "[masking]")
{
    CHECK(IsKeptStatic(static_cast<std::uint8_t>(kKeptStatic)));
    CHECK_FALSE(IsKeptStatic(static_cast<std::uint8_t>(kEgoRoi)));
    CHECK(IsDynamic(static_cast<std::uint8_t>(kDynamic2D | raylidar::masking::kDynamicUnion)));
    CHECK_FALSE(IsDynamic(static_cast<std::uint8_t>(kKeptStatic)));
    CHECK(IsEgo(static_cast<std::uint8_t>(kEgoRoi)));
    CHECK_FALSE(IsEgo(static_cast<std::uint8_t>(kDynamicBBox)));
}
