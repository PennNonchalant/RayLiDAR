#include "fixture.h"

#include <raylidar/common/io.h>
#include <raylidar/lidar/uncurl.h>
#include <raylidar/lidar/velodyne.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace
{
using Catch::Matchers::WithinAbs;
using raylidar::IoError;
using raylidar::lidar::AzimuthFraction;
using raylidar::lidar::InterpolateRigid;
using raylidar::lidar::PointXYZI;
using raylidar::lidar::UncurlParams;
using raylidar::lidar::UncurlPoint;
using raylidar::lidar::UncurlScan;

constexpr double kTolerance = 1e-9;

Eigen::Matrix4d Translation(double x, double y, double z)
{
    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
    transform(0, 3) = x;
    transform(1, 3) = y;
    transform(2, 3) = z;
    return transform;
}

Eigen::Matrix4d Yaw(double angle)
{
    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    transform(0, 0) = c;
    transform(0, 1) = -s;
    transform(1, 0) = s;
    transform(1, 1) = c;
    return transform;
}

UncurlParams Params(const Eigen::Matrix4d &current, const Eigen::Matrix4d &next)
{
    UncurlParams params;
    params.T_W_V_curr = current;
    params.T_W_V_next = next;
    params.t_curr_ns = 0;
    params.t_next_ns = 1000000000;
    return params;
}

void CheckVector(const Eigen::Vector3d &actual, const Eigen::Vector3d &expected)
{
    CHECK_THAT(actual.x(), WithinAbs(expected.x(), kTolerance));
    CHECK_THAT(actual.y(), WithinAbs(expected.y(), kTolerance));
    CHECK_THAT(actual.z(), WithinAbs(expected.z(), kTolerance));
}
} // namespace

TEST_CASE("AzimuthFraction follows the devkit 0.5 + atan2/2pi convention", "[lidar]")
{
    CHECK_THAT(AzimuthFraction({10.0, 0.0, 0.0}), WithinAbs(0.5, kTolerance));
    CHECK_THAT(AzimuthFraction({0.0, 10.0, 0.0}), WithinAbs(0.75, kTolerance));
    CHECK_THAT(AzimuthFraction({0.0, -10.0, 0.0}), WithinAbs(0.25, kTolerance));
    CHECK_THAT(AzimuthFraction({-10.0, 0.0, 0.0}), WithinAbs(0.0, kTolerance));

    const double positivePi = AzimuthFraction({-10.0, 0.0, 0.0});
    CHECK(positivePi >= 0.0);
    CHECK(positivePi < 1.0);
}

TEST_CASE("Zero ego motion returns the input point unchanged", "[lidar]")
{
    const UncurlParams params = Params(Eigen::Matrix4d::Identity(), Eigen::Matrix4d::Identity());
    const Eigen::Vector3d point(12.5, -3.0, 1.25);
    CheckVector(UncurlPoint(point, params), point);
    CheckVector(UncurlPoint({0.0, 10.0, 0.0}, params), {0.0, 10.0, 0.0});
}

TEST_CASE("Constant-velocity straight-line motion shifts points by u * v * dt", "[lidar]")
{
    // 1 m displacement over one 1 s sweep, so u in [0, 1) maps to [0, 1) metres.
    const UncurlParams params =
        Params(Eigen::Matrix4d::Identity(), Translation(1.0, 0.0, 0.0));

    CheckVector(UncurlPoint({10.0, 0.0, 0.0}, params), {10.5, 0.0, 0.0});
    CheckVector(UncurlPoint({0.0, 10.0, 0.0}, params), {0.75, 10.0, 0.0});
    CheckVector(UncurlPoint({0.0, -10.0, 0.0}, params), {0.25, -10.0, 0.0});
    CheckVector(UncurlPoint({-10.0, 0.0, 0.0}, params), {-9.0, 0.0, 0.0});
}

TEST_CASE("A known yaw rotation produces the analytically expected rotation", "[lidar]")
{
    // Next pose is a 0.2 rad yaw; a point at u = 0.5 sees 0.1 rad.
    const UncurlParams params =
        Params(Eigen::Matrix4d::Identity(), Yaw(0.2));

    const Eigen::Vector3d atHalf(10.0, 0.0, 0.0);
    const Eigen::Vector3d expectedHalf = Yaw(0.1) * atHalf;
    CheckVector(UncurlPoint(atHalf, params), expectedHalf);

    const Eigen::Vector3d atThreeQuarters(0.0, 10.0, 0.0);
    const Eigen::Vector3d expectedThreeQuarters = Yaw(0.15) * atThreeQuarters;
    CheckVector(UncurlPoint(atThreeQuarters, params), expectedThreeQuarters);
}

TEST_CASE("InterpolateRigid reproduces both endpoints and is finite", "[lidar]")
{
    const Eigen::Matrix4d start = Translation(1.0, 2.0, 3.0);
    const Eigen::Matrix4d end = Translation(4.0, 6.0, 3.0) * Yaw(0.4);

    const Eigen::Matrix4d atZero = InterpolateRigid(start, end, 0.0);
    const Eigen::Matrix4d atOne = InterpolateRigid(start, end, 1.0);
    CHECK((atZero - start).cwiseAbs().maxCoeff() < 1e-9);
    CHECK((atOne - end).cwiseAbs().maxCoeff() < 1e-9);

    const Eigen::Matrix4d middle = InterpolateRigid(start, end, 0.5);
    CHECK(middle.allFinite());
    CHECK_THAT(middle(0, 3), WithinAbs(2.5, 1e-9));
    CHECK_THAT(middle(1, 3), WithinAbs(4.0, 1e-9));
}

TEST_CASE("UncurlScan rejects non-finite points with a counted reason", "[lidar]")
{
    const UncurlParams params =
        Params(Eigen::Matrix4d::Identity(), Translation(1.0, 0.0, 0.0));
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const std::vector<PointXYZI> points = {
        {10.0F, 0.0F, 0.0F, 1.0F},
        {static_cast<float>(nan), 0.0F, 0.0F, 2.0F},
        {0.0F, 10.0F, 0.0F, 3.0F},
    };

    std::size_t rejected = 0;
    const std::vector<Eigen::Vector3d> uncurled = UncurlScan(points, params, &rejected);
    REQUIRE(uncurled.size() == 2U);
    CHECK(rejected == 1U);
    CheckVector(uncurled[0], {10.5, 0.0, 0.0});
    CheckVector(uncurled[1], {0.75, 10.0, 0.0});

    CHECK_THROWS_AS(UncurlPoint({nan, 0.0, 0.0}, params), std::invalid_argument);
}

TEST_CASE("UncurlScan rejects a non-positive sweep interval", "[lidar]")
{
    UncurlParams params;
    params.t_curr_ns = 2000000000;
    params.t_next_ns = 2000000000;
    CHECK_THROWS_AS(UncurlScan({{1.0F, 0.0F, 0.0F, 0.0F}}, params), IoError);

    params.t_next_ns = 1000000000;
    CHECK_THROWS_AS(UncurlScan({{1.0F, 0.0F, 0.0F, 0.0F}}, params), IoError);
}

TEST_CASE("InterpolateRigid rejects non-finite transforms", "[lidar]")
{
    Eigen::Matrix4d broken = Eigen::Matrix4d::Identity();
    broken(0, 3) = std::numeric_limits<double>::infinity();
    CHECK_THROWS_AS(InterpolateRigid(broken, Eigen::Matrix4d::Identity(), 0.5),
                    std::invalid_argument);
}
