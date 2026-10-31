#include "fixture.h"

#include <raylidar/gt/camera_assign.h>

#include <catch2/catch_test_macros.hpp>

#include <cmath>

namespace
{
using raylidar::gt::AssignPrimaryCamera;
using raylidar::gt::CameraEligibility;
using raylidar::gt::EvaluateCameraEligibility;
using raylidar::test::MakeTwoCameraRig;
using raylidar::test::Translate;

constexpr double kTolerance = 1e-9;

CameraEligibility Evaluate(const Eigen::Vector3d &point, const Eigen::Matrix4d &transform,
                           const Eigen::Matrix3d &intrinsics, const int width, const int height)
{
    return EvaluateCameraEligibility(point, transform, intrinsics, width, height, 1.5, 60.0, 5);
}
} // namespace

TEST_CASE("a point valid in both cameras is assigned to the smaller normalized radius", "[gt]")
{
    TwoCameraRig rig = MakeTwoCameraRig();
    rig.T_Ck_rect_V["image_00"] = Translate(2.0, 0.0, 0.0);
    rig.T_Ck_rect_V["image_01"] = Translate(1.0, 0.0, 0.0);

    const auto assignment =
        AssignPrimaryCamera(Eigen::Vector3d(0.0, 0.0, 10.0), rig.T_Ck_rect_V, rig.K_rect,
                            rig.imageSizes, 1.5, 60.0, 5);
    CHECK(assignment.camera == 1U);
    CHECK(std::abs(assignment.r_gt - std::sqrt(101.0)) < kTolerance);
    CHECK(std::abs(assignment.u - 210.0) < kTolerance);
    CHECK(std::abs(assignment.v - 150.0) < kTolerance);

    rig.T_Ck_rect_V["image_00"] = Translate(1.0, 0.0, 0.0);
    rig.T_Ck_rect_V["image_01"] = Translate(2.0, 0.0, 0.0);
    const auto flipped =
        AssignPrimaryCamera(Eigen::Vector3d(0.0, 0.0, 10.0), rig.T_Ck_rect_V, rig.K_rect,
                            rig.imageSizes, 1.5, 60.0, 5);
    CHECK(flipped.camera == 0U);
    CHECK(std::abs(flipped.r_gt - std::sqrt(101.0)) < kTolerance);
}

TEST_CASE("an exact normalized-radius tie chooses image_00", "[gt]")
{
    const TwoCameraRig rig = MakeTwoCameraRig();
    const auto assignment =
        AssignPrimaryCamera(Eigen::Vector3d(0.0, 0.0, 10.0), rig.T_Ck_rect_V, rig.K_rect,
                            rig.imageSizes, 1.5, 60.0, 5);
    CHECK(assignment.camera == 0U);
    CHECK(std::abs(assignment.u - 200.0) < kTolerance);
    CHECK(std::abs(assignment.v - 150.0) < kTolerance);
}

TEST_CASE("a point behind one camera uses the other valid camera", "[gt]")
{
    TwoCameraRig rig = MakeTwoCameraRig();
    rig.T_Ck_rect_V["image_00"] = Translate(0.0, 0.0, -20.0);

    const auto behind = Evaluate(Eigen::Vector3d(0.0, 0.0, 10.0), rig.T_Ck_rect_V["image_00"],
                                 rig.K_rect["image_00"], 400, 300);
    CHECK_FALSE(behind.zPositive);
    CHECK(behind.rangeOk);

    const auto assignment =
        AssignPrimaryCamera(Eigen::Vector3d(0.0, 0.0, 10.0), rig.T_Ck_rect_V, rig.K_rect,
                            rig.imageSizes, 1.5, 60.0, 5);
    CHECK(assignment.camera == 1U);
    CHECK(assignment.r_gt > 0.0);
}

TEST_CASE("a point outside every camera's FOV is no_camera_fov", "[gt]")
{
    TwoCameraRig rig = MakeTwoCameraRig();
    rig.T_Ck_rect_V["image_00"] = Translate(0.0, 0.0, -20.0);
    rig.T_Ck_rect_V["image_01"] = Translate(0.0, 0.0, -20.0);

    const auto behind = Evaluate(Eigen::Vector3d(0.0, 0.0, 10.0), rig.T_Ck_rect_V["image_00"],
                                 rig.K_rect["image_00"], 400, 300);
    CHECK_FALSE(behind.zPositive);
    CHECK_FALSE(behind.inImageMargin);
    CHECK(behind.rangeOk);

    const auto assignment =
        AssignPrimaryCamera(Eigen::Vector3d(0.0, 0.0, 10.0), rig.T_Ck_rect_V, rig.K_rect,
                            rig.imageSizes, 1.5, 60.0, 5);
    CHECK(assignment.camera == 255U);
    CHECK(assignment.r_gt == -1.0);

    TwoCameraRig outside = MakeTwoCameraRig();
    const auto margin = Evaluate(Eigen::Vector3d(20.0, 0.0, 10.0), outside.T_Ck_rect_V["image_00"],
                                 outside.K_rect["image_00"], 400, 300);
    CHECK(margin.zPositive);
    CHECK_FALSE(margin.inImageMargin);
    CHECK(margin.rangeOk);
}

TEST_CASE("an in-FOV point outside [1.5,60] m reports range_out_of_domain", "[gt]")
{
    const TwoCameraRig rig = MakeTwoCameraRig();

    const auto near = Evaluate(Eigen::Vector3d(0.0, 0.0, 0.5), rig.T_Ck_rect_V["image_00"],
                               rig.K_rect["image_00"], 400, 300);
    CHECK(near.zPositive);
    CHECK(near.inImageMargin);
    CHECK_FALSE(near.rangeOk);
    CHECK_FALSE(near.valid());

    const auto assignment =
        AssignPrimaryCamera(Eigen::Vector3d(0.0, 0.0, 0.5), rig.T_Ck_rect_V, rig.K_rect,
                            rig.imageSizes, 1.5, 60.0, 5);
    CHECK(assignment.camera == 255U);
    CHECK(assignment.r_gt == -1.0);

    const auto far = Evaluate(Eigen::Vector3d(0.0, 0.0, 80.0), rig.T_Ck_rect_V["image_00"],
                              rig.K_rect["image_00"], 400, 300);
    CHECK(far.zPositive);
    CHECK(far.inImageMargin);
    CHECK_FALSE(far.rangeOk);
}
