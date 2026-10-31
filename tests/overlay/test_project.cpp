#include "fixture.h"

#include <raylidar/overlay/distance_transform.h>
#include <raylidar/overlay/project.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
namespace fs = std::filesystem;
using Catch::Approx;
using raylidar::overlay::EdgeDistanceScore;
using raylidar::overlay::EuclideanDistanceTransform;
using raylidar::overlay::OverlayFrame;
using raylidar::overlay::ProjectionResult;
using raylidar::overlay::ProjectPoints;
using raylidar::overlay::RenderOverlay;
using raylidar::test::GrayImageWithBlackRect;
using raylidar::test::MakeOverlayTempRoot;
using raylidar::test::OverlayIntrinsics;
using raylidar::test::RgbImageWithBlackRect;
using raylidar::test::Translation4;
using raylidar::test::WritePngWithDimensions;
using raylidar::test::WriteRgbPng;

constexpr int kWidth = 1408;
constexpr int kHeight = 376;

Eigen::MatrixXd Points(std::initializer_list<Eigen::Vector3d> values)
{
    Eigen::MatrixXd points(static_cast<Eigen::Index>(values.size()), 3);
    Eigen::Index row = 0;
    for (const Eigen::Vector3d &value : values)
    {
        points.row(row++) = value.transpose();
    }
    return points;
}

ProjectionResult ManualResult(int width, int height, const std::vector<Eigen::Vector2d> &uv,
                              const std::vector<double> &z)
{
    ProjectionResult result;
    result.imageWidth = width;
    result.imageHeight = height;
    for (std::size_t index = 0; index < uv.size(); ++index)
    {
        result.uv.push_back(uv[index]);
        result.z.push_back(z[index]);
        result.rangeM.push_back(z[index]);
        result.valid.push_back(true);
        result.sourceIndex.push_back(index);
    }
    return result;
}
} // namespace

TEST_CASE("ProjectPoints maps axis points to exact pixels", "[overlay]")
{
    const Eigen::Matrix3d K = OverlayIntrinsics(700.0, 700.0, 704.0, 188.0);
    const Eigen::Matrix4d identity = Eigen::Matrix4d::Identity();
    const Eigen::MatrixXd points =
        Points({{0.0, 0.0, 10.0}, {0.0, 0.0, 20.0}, {0.0, 0.0, 30.0}, {1.0, 0.0, 10.0},
                {0.0, 0.5, 20.0}});
    const ProjectionResult result =
        ProjectPoints(points, identity, K, kWidth, kHeight, 10.0, 30.0, 5, true);

    REQUIRE(result.uv.size() == 5U);
    for (std::size_t index = 0; index < result.valid.size(); ++index)
    {
        CHECK(result.valid[index]);
        CHECK(result.sourceIndex[index] == index);
    }
    CHECK(result.uv[0].x() == Approx(704.0).margin(1e-9));
    CHECK(result.uv[0].y() == Approx(188.0).margin(1e-9));
    CHECK(result.uv[1].x() == Approx(704.0).margin(1e-9));
    CHECK(result.uv[3].x() == Approx(774.0).margin(1e-9));
    CHECK(result.uv[4].y() == Approx(205.5).margin(1e-9));
    CHECK(result.rangeM[0] == Approx(10.0).margin(1e-12));
    CHECK(result.rangeM[1] == Approx(20.0).margin(1e-12));
    CHECK(result.rangeM[2] == Approx(30.0).margin(1e-12));
    CHECK(result.z[2] == Approx(30.0).margin(1e-12));
}

TEST_CASE("ProjectPoints rejects points behind the camera", "[overlay]")
{
    const Eigen::Matrix3d K = OverlayIntrinsics(700.0, 700.0, 704.0, 188.0);
    const Eigen::Matrix4d identity = Eigen::Matrix4d::Identity();
    const Eigen::MatrixXd points = Points({{0.0, 0.0, -1.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 10.0}});
    const ProjectionResult result =
        ProjectPoints(points, identity, K, kWidth, kHeight, 10.0, 30.0, 5, true);

    CHECK_FALSE(result.valid[0]);
    CHECK_FALSE(result.valid[1]);
    CHECK(result.valid[2]);
    CHECK(std::isnan(result.uv[0].x()));
}

TEST_CASE("ProjectPoints excludes pixels inside the margin", "[overlay]")
{
    const Eigen::Matrix3d K = OverlayIntrinsics(700.0, 700.0, 704.0, 188.0);
    const Eigen::Matrix4d identity = Eigen::Matrix4d::Identity();
    const double z = 20.0;
    const double xEdge = (3.0 - 704.0) * z / 700.0;
    const double xInside = (6.0 - 704.0) * z / 700.0;
    const double yEdge = (3.0 - 188.0) * z / 700.0;
    const double yInside = (6.0 - 188.0) * z / 700.0;
    const Eigen::MatrixXd points = Points({{xEdge, 0.0, z},
                                           {xInside, 0.0, z},
                                           {0.0, yEdge, z},
                                           {0.0, yInside, z}});
    const ProjectionResult result =
        ProjectPoints(points, identity, K, kWidth, kHeight, 10.0, 30.0, 5, true);

    CHECK(result.uv[0].x() == Approx(3.0).margin(1e-9));
    CHECK(result.uv[1].x() == Approx(6.0).margin(1e-9));
    CHECK(result.uv[2].y() == Approx(3.0).margin(1e-9));
    CHECK(result.uv[3].y() == Approx(6.0).margin(1e-9));
    CHECK_FALSE(result.valid[0]);
    CHECK(result.valid[1]);
    CHECK_FALSE(result.valid[2]);
    CHECK(result.valid[3]);
}

TEST_CASE("ProjectPoints uses the camera-frame 10-30 m band", "[overlay]")
{
    const Eigen::Matrix3d K = OverlayIntrinsics(700.0, 700.0, 704.0, 188.0);
    const Eigen::Matrix4d identity = Eigen::Matrix4d::Identity();
    const Eigen::MatrixXd points = Points({{0.0, 0.0, 5.0}, {0.0, 0.0, 40.0}});

    const ProjectionResult clamped =
        ProjectPoints(points, identity, K, kWidth, kHeight, 10.0, 30.0, 5, true);
    CHECK_FALSE(clamped.valid[0]);
    CHECK_FALSE(clamped.valid[1]);

    const ProjectionResult unclamped =
        ProjectPoints(points, identity, K, kWidth, kHeight, 10.0, 30.0, 5, false);
    CHECK(unclamped.valid[0]);
    CHECK(unclamped.valid[1]);
}

TEST_CASE("ProjectPoints rangeM is measured in the camera frame", "[overlay]")
{
    const Eigen::Matrix3d K = OverlayIntrinsics(700.0, 700.0, 704.0, 188.0);
    const Eigen::Matrix4d offset = Translation4(5.0, 0.0, 0.0);
    const Eigen::MatrixXd points = Points({{0.0, 0.0, 20.0}});
    const ProjectionResult result =
        ProjectPoints(points, offset, K, kWidth, kHeight, 10.0, 30.0, 5, true);

    REQUIRE(result.valid[0]);
    CHECK(result.rangeM[0] == Approx(std::sqrt(425.0)).margin(1e-12));
    CHECK(result.rangeM[0] != Approx(20.0).margin(1e-6));
    CHECK(result.uv[0].x() == Approx(700.0 * 5.0 / 20.0 + 704.0).margin(1e-9));
}

TEST_CASE("EdgeDistanceScore is small on a rectangle edge", "[overlay]")
{
    const int width = 256;
    const int height = 128;
    const std::vector<std::uint8_t> gray = GrayImageWithBlackRect(width, height, 80, 40, 176, 88);

    std::vector<Eigen::Vector2d> uv;
    std::vector<double> z;
    for (int index = 0; index < 60; ++index)
    {
        uv.emplace_back(90.0 + static_cast<double>(index), 40.0);
        z.push_back(20.0);
    }
    const ProjectionResult result = ManualResult(width, height, uv, z);
    const nlohmann::json score = EdgeDistanceScore(gray, width, height, result);

    CHECK(score["count"] == 60);
    CHECK(score["median_px"].get<double>() <= 1.5);
    CHECK(score["fraction_within_2px"].get<double>() >= 0.99);
}

TEST_CASE("EdgeDistanceScore prefers depth discontinuities near edges", "[overlay]")
{
    const int width = 256;
    const int height = 128;
    const std::vector<std::uint8_t> gray = GrayImageWithBlackRect(width, height, 80, 40, 176, 88);

    std::vector<Eigen::Vector2d> nearUv;
    std::vector<Eigen::Vector2d> farUv;
    std::vector<double> nearZ;
    std::vector<double> farZ;
    for (int index = 0; index < 60; ++index)
    {
        const double x = 90.0 + static_cast<double>(index);
        const double depth = (index % 2 == 0) ? 20.0 : 25.0;
        nearUv.emplace_back(x, 40.0);
        nearZ.push_back(depth);
        farUv.emplace_back(x, 10.0);
        farZ.push_back(depth);
    }

    const nlohmann::json nearScore =
        EdgeDistanceScore(gray, width, height, ManualResult(width, height, nearUv, nearZ));
    const nlohmann::json farScore =
        EdgeDistanceScore(gray, width, height, ManualResult(width, height, farUv, farZ));

    CHECK_FALSE(nearScore["fallback_all_points"].get<bool>());
    CHECK_FALSE(farScore["fallback_all_points"].get<bool>());
    CHECK(nearScore["median_px"].get<double>() < farScore["median_px"].get<double>());
    CHECK(farScore["median_px"].get<double>() > 20.0);
}

TEST_CASE("EdgeDistanceScore handles empty input without throwing", "[overlay]")
{
    const std::vector<std::uint8_t> gray = GrayImageWithBlackRect(16, 16, 4, 4, 8, 8);
    const ProjectionResult empty;
    const nlohmann::json score = EdgeDistanceScore(gray, 16, 16, empty);

    CHECK(score["count"] == 0);
    CHECK(std::isnan(score["median_px"].get<double>()));
    CHECK(std::isnan(score["p90_px"].get<double>()));
}

TEST_CASE("EuclideanDistanceTransform is exact", "[overlay]")
{
    std::vector<std::uint8_t> binary(9U, 0U);
    binary[4] = 1U;
    const std::vector<double> distance = EuclideanDistanceTransform(binary, 3, 3);
    REQUIRE(distance.size() == 9U);
    CHECK(distance[4] == Approx(0.0).margin(1e-12));
    CHECK(distance[1] == Approx(1.0).margin(1e-12));
    CHECK(distance[3] == Approx(1.0).margin(1e-12));
    CHECK(distance[0] == Approx(std::sqrt(2.0)).margin(1e-12));

    const std::vector<double> empty = EuclideanDistanceTransform(std::vector<std::uint8_t>(9U, 0U), 3, 3);
    for (const double value : empty)
    {
        CHECK(std::isnan(value));
    }
}

TEST_CASE("OverlayFrame writes exact per-camera artifact paths", "[overlay]")
{
    const fs::path root = MakeOverlayTempRoot("paths");
    const int width = 64;
    const int height = 48;
    const std::vector<std::uint8_t> rgb = RgbImageWithBlackRect(width, height, 20, 15, 44, 33);
    const fs::path rectified = root / "data_2d_raw" / "seq" / "image_00" / "data_rect" /
                               "0000000000.png";
    WriteRgbPng(rectified, width, height, rgb);
    const fs::path raw = root / "data_2d_raw" / "seq" / "image_00" / "data_rgb" /
                         "0000000000.png";
    WriteRgbPng(raw, width, height, rgb);

    const Eigen::Matrix3d K = OverlayIntrinsics(50.0, 50.0, 32.0, 24.0);
    const Eigen::MatrixXd points = Points({{0.0, 0.0, 20.0}, {0.5, 0.2, 22.0}});
    const fs::path outDir = root / "out";

    const nlohmann::json report =
        OverlayFrame(rectified, points, Eigen::Matrix4d::Identity(), K, outDir, 0, "image_00", 4);
    const fs::path frameDir = outDir / "frame_0000000000";
    CHECK(fs::is_regular_file(frameDir / "image_00_overlay_1x.png"));
    CHECK(fs::is_regular_file(frameDir / "image_00_overlay_4x.png"));
    CHECK(report["image"] == rectified.string());
    CHECK(report["image_resolved"] == fs::weakly_canonical(rectified).string());
    CHECK(report["rectified"] == true);
    CHECK(report["image_size"] == nlohmann::json::array({width, height}));
    CHECK(report["scale_factor"] == 4);
    CHECK(report["overlay_1x"] == (frameDir / "image_00_overlay_1x.png").string());
    CHECK(report["overlay_4x"] == (frameDir / "image_00_overlay_4x.png").string());
    CHECK(report["points_total"] == 2);
    CHECK(report["points_valid"] == 2);
    CHECK(report["projection_invariant"] == true);

    int decodedWidth = 0;
    int decodedHeight = 0;
    const std::vector<std::uint8_t> decoded =
        raylidar::test::ReadRgbPng(frameDir / "image_00_overlay_1x.png", decodedWidth,
                                   decodedHeight);
    CHECK(decodedWidth == width);
    CHECK(decodedHeight == height);
    CHECK(decoded.size() == static_cast<std::size_t>(width * height * 3));

    const nlohmann::json rawReport =
        OverlayFrame(raw, points, Eigen::Matrix4d::Identity(), K, outDir, 0, "image_00", 4);
    CHECK(rawReport["rectified"] == false);
}

TEST_CASE("OverlayFrame rejects oversized images before allocation", "[overlay]")
{
    const fs::path root = MakeOverlayTempRoot("too_large");
    const fs::path huge = root / "data_rect" / "0000000000.png";
    WritePngWithDimensions(huge, 8000, 8000);

    const Eigen::MatrixXd points(0, 3);
    CHECK_THROWS_AS(OverlayFrame(huge, points, Eigen::Matrix4d::Identity(),
                                 OverlayIntrinsics(50.0, 50.0, 32.0, 24.0), root / "out", 0,
                                 "image_00"),
                    std::invalid_argument);
}

TEST_CASE("RenderOverlay still returns the full image without valid points", "[overlay]")
{
    const int width = 32;
    const int height = 16;
    const std::vector<std::uint8_t> rgb = RgbImageWithBlackRect(width, height, 8, 4, 16, 12);
    const ProjectionResult empty;
    const std::vector<std::uint8_t> rendered = RenderOverlay(rgb, width, height, empty, 2, 1, 80);
    CHECK(rendered.size() == static_cast<std::size_t>(width * height * 4 * 3));
}
