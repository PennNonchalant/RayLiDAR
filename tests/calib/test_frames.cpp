#include <raylidar/calib/frames.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <map>
#include <stdexcept>
#include <vector>

namespace
{
using raylidar::calib::AssertRotation;
using raylidar::calib::AssertRoundtrip;
using raylidar::calib::CameraHeight;
using raylidar::calib::Compose;
using raylidar::calib::EstimateGroundPlane;
using raylidar::calib::GroundPlane;
using raylidar::calib::InvertRigid;
using raylidar::calib::LookupPose;
using raylidar::calib::MakeSegTransform;
using raylidar::calib::PoseLookup;
using raylidar::calib::RotationOf;
using raylidar::calib::TranslationOf;

constexpr double kPi = 3.14159265358979323846;

Eigen::Matrix4d Rigid(const Eigen::Matrix3d &rotation, const Eigen::Vector3d &translation)
{
    Eigen::Matrix4d matrix = Eigen::Matrix4d::Identity();
    matrix.topLeftCorner<3, 3>() = rotation;
    matrix.topRightCorner<3, 1>() = translation;
    return matrix;
}

Eigen::Matrix3d Rotation(double angle, const Eigen::Vector3d &axis)
{
    return Eigen::AngleAxisd(angle, axis.normalized()).toRotationMatrix();
}

Eigen::Matrix3d NearOrthonormal(double perturbation)
{
    Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
    rotation(0, 1) = perturbation;
    rotation(1, 0) = perturbation;
    return rotation;
}

Eigen::MatrixXd GroundPoints(bool sloped)
{
    constexpr int kGround = 500;
    constexpr int kWalls = 200;
    const int rows = sloped ? kGround + kWalls : kGround;
    Eigen::MatrixXd points(rows, 3);
    for (int index = 0; index < kGround; ++index)
    {
        const double angle = 2.0 * kPi * static_cast<double>(index) / kGround;
        const double radius = 5.0 + 2.5 * static_cast<double>(index % 10);
        const double x = radius * std::cos(angle);
        const double y = radius * std::sin(angle);
        points(index, 0) = x;
        points(index, 1) = y;
        points(index, 2) = sloped ? 0.02 * x - 1.80 : -1.80;
    }
    if (sloped)
    {
        for (int index = 0; index < kWalls; ++index)
        {
            const double angle = 2.0 * kPi * static_cast<double>(index) / kWalls;
            const double radius = 6.0 + 2.0 * static_cast<double>(index % 12);
            const double x = radius * std::cos(angle);
            const double y = radius * std::sin(angle);
            points(kGround + index, 0) = x;
            points(kGround + index, 1) = y;
            points(kGround + index, 2) = 0.5 + 1.5 * static_cast<double>(index % 7) / 7.0;
        }
    }
    return points;
}
} // namespace

TEST_CASE("Compose applies the destination-source naming rule", "[calib]")
{
    const Eigen::Matrix4d T_A_B = Rigid(Rotation(0.4, Eigen::Vector3d(1.0, 2.0, 3.0)), Eigen::Vector3d(1.0, -2.0, 0.5));
    const Eigen::Matrix4d T_B_C =
        Rigid(Rotation(-0.9, Eigen::Vector3d(-2.0, 1.0, 0.5)), Eigen::Vector3d(0.25, 4.0, -1.0));
    const Eigen::Matrix4d composed = Compose(T_A_B, T_B_C);
    const Eigen::Vector4d point(0.7, -0.3, 2.0, 1.0);

    const Eigen::Vector4d expected = T_A_B * (T_B_C * point);
    CHECK(((composed * point) - expected).norm() == Catch::Approx(0.0).margin(1e-12));

    const Eigen::Matrix4d identity = Compose(T_A_B, InvertRigid(T_A_B));
    CHECK((identity - Eigen::Matrix4d::Identity()).cwiseAbs().maxCoeff() <= 1e-12);
}

TEST_CASE("InvertRigid tolerates official-file precision", "[calib]")
{
    SECTION("1e-6 perturbed rotation is accepted")
    {
        const Eigen::Matrix4d transform = Rigid(NearOrthonormal(5e-7), Eigen::Vector3d(1.0, 2.0, 3.0));
        Eigen::Matrix4d inverse;
        CHECK_NOTHROW(inverse = InvertRigid(transform));
        const Eigen::Matrix4d product = transform * inverse;
        CHECK((product - Eigen::Matrix4d::Identity()).cwiseAbs().maxCoeff() <= 1e-5);
    }

    SECTION("kilometre-scale transform uses a scale-aware tolerance")
    {
        const Eigen::Matrix4d transform = Rigid(NearOrthonormal(5e-7), Eigen::Vector3d(4000.0, 10.0, 5.0));
        CHECK_NOTHROW(InvertRigid(transform));
    }

    SECTION("1e-2 perturbed rotation is rejected")
    {
        const Eigen::Matrix4d transform = Rigid(NearOrthonormal(1e-2), Eigen::Vector3d(1.0, 2.0, 3.0));
        CHECK_THROWS(InvertRigid(transform));
    }
}

TEST_CASE("AssertRoundtrip gates camera-relative transforms at 1e-9", "[calib]")
{
    const Eigen::Matrix4d T_Ck_rect_V =
        Rigid(Rotation(0.7, Eigen::Vector3d(0.2, -1.0, 0.4)), Eigen::Vector3d(0.4, 0.1, 0.8));
    CHECK_NOTHROW(AssertRoundtrip(T_Ck_rect_V, InvertRigid(T_Ck_rect_V), 1e-9));

    Eigen::Matrix4d perturbed = InvertRigid(T_Ck_rect_V);
    perturbed(0, 3) += 1e-6;
    CHECK_THROWS(AssertRoundtrip(T_Ck_rect_V, perturbed, 1e-9));
}

TEST_CASE("AssertRotation rejects scaled and reflected matrices", "[calib]")
{
    CHECK_NOTHROW(AssertRotation(NearOrthonormal(5e-7)));

    Eigen::Matrix3d scaled = Eigen::Matrix3d::Identity() * 1.001;
    CHECK_THROWS(AssertRotation(scaled));

    Eigen::Matrix3d reflected = Eigen::Matrix3d::Identity();
    reflected(2, 2) = -1.0;
    CHECK_THROWS(AssertRotation(reflected));

    CHECK_THROWS(AssertRotation(NearOrthonormal(1e-3)));
}

TEST_CASE("LookupPose never falls back to frame 0", "[calib]")
{
    const std::map<int, Eigen::Matrix4d> poses = {
        {0, Rigid(Eigen::Matrix3d::Identity(), Eigen::Vector3d(0.0, 0.0, 0.0))},
        {5, Rigid(Eigen::Matrix3d::Identity(), Eigen::Vector3d(5.0, 0.0, 0.0))},
        {10, Rigid(Eigen::Matrix3d::Identity(), Eigen::Vector3d(10.0, 0.0, 0.0))},
    };

    const PoseLookup exact = LookupPose(poses, 5, 5);
    CHECK(exact.poseFrame == 5);
    CHECK(exact.delta == 0);

    const std::map<int, Eigen::Matrix4d> tie = {
        {0, Rigid(Eigen::Matrix3d::Identity(), Eigen::Vector3d(0.0, 0.0, 0.0))},
        {2, Rigid(Eigen::Matrix3d::Identity(), Eigen::Vector3d(2.0, 0.0, 0.0))},
    };
    const PoseLookup nearest = LookupPose(tie, 1, 5);
    CHECK(nearest.poseFrame == 0);
    CHECK(nearest.delta == -1);

    CHECK_THROWS_AS(LookupPose(poses, 8, 1), std::out_of_range);
    // Nearest pose (10) is farther than maxDelta; must throw, not return frame 0.
    CHECK_THROWS_AS(LookupPose(poses, 16, 5), std::out_of_range);
}

TEST_CASE("MakeSegTransform anchors the median camera centre at the origin", "[calib]")
{
    const std::vector<Eigen::Vector3d> centres = {
        Eigen::Vector3d(0.0, 0.0, 0.0),
        Eigen::Vector3d(2.0, 4.0, 6.0),
        Eigen::Vector3d(4.0, 8.0, 12.0),
    };
    const auto [T_S_W, T_W_S] = MakeSegTransform(centres);
    // Component-wise median of {(0,0,0), (2,4,6), (4,8,12)} is (2,4,6).
    const Eigen::Vector4d median(2.0, 4.0, 6.0, 1.0);
    CHECK(((T_S_W * median).head<3>()).norm() <= 1e-9);
    CHECK((T_W_S * Eigen::Vector4d(0.0, 0.0, 0.0, 1.0)).head<3>().isApprox(Eigen::Vector3d(2.0, 4.0, 6.0), 1e-9));
}

TEST_CASE("EstimateGroundPlane returns a perpendicular camera height", "[calib]")
{
    const Eigen::Matrix4d T_V_C0 = Rigid(Eigen::Matrix3d::Identity(), Eigen::Vector3d(0.10, 0.0, -0.30));

    SECTION("flat ground gives 1.50 m")
    {
        const GroundPlane plane = EstimateGroundPlane(GroundPoints(false));
        const double height = CameraHeight(T_V_C0, plane);
        CHECK(height == Catch::Approx(1.50).margin(1e-6));
    }

    SECTION("sloped ground with wall outliers is not pulled")
    {
        const GroundPlane plane = EstimateGroundPlane(GroundPoints(true));
        const double height = CameraHeight(T_V_C0, plane);
        CHECK(height == Catch::Approx(1.50).margin(0.01));
        CHECK(height < 1.6);
    }

    SECTION("too few inliers throws")
    {
        Eigen::MatrixXd tiny(10, 3);
        for (int index = 0; index < 10; ++index)
        {
            tiny(index, 0) = 5.0 + static_cast<double>(index);
            tiny(index, 1) = 0.0;
            tiny(index, 2) = -1.8;
        }
        CHECK_THROWS_AS(EstimateGroundPlane(tiny), std::invalid_argument);
    }
}
