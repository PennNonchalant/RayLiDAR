#include "fixture.h"

#include <raylidar/calib/frames.h>
#include <raylidar/calib/kitti360.h>
#include <raylidar/common/io.h>

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace
{
namespace fs = std::filesystem;
using raylidar::IoError;
using raylidar::calib::Compose;
using raylidar::calib::ComposeTCkRectV;
using raylidar::calib::ComposeTWCkRect;
using raylidar::calib::ComposeTWV;
using raylidar::calib::FindCommonFrames;
using raylidar::calib::InvertRigid;
using raylidar::calib::Kitti360Calib;
using raylidar::calib::LoadCam0ToWorld;
using raylidar::calib::LoadKitti360Calib;
using raylidar::calib::LoadPoses;
using raylidar::calib::LookupPose;
using raylidar::calib::Rect4;
using raylidar::calib::TranslationOf;
using raylidar::test::CalibTreeOptions;
using raylidar::test::MakeCalibTree;
using raylidar::test::MakeTempRoot;
using raylidar::test::ReadFixtureFile;

constexpr const char *kSequence = "2013_05_28_drive_0000_sync";
const std::vector<std::string> kCameras = {"image_00", "image_01"};

Kitti360Calib LoadFixture(const fs::path &root)
{
    return LoadKitti360Calib(root / "calibration");
}

void WriteTextFile(const fs::path &path, const std::string &text)
{
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    REQUIRE(output.good());
    output << text;
}

double MaxAbsDiff(const Eigen::Matrix4d &a, const Eigen::Matrix4d &b)
{
    return (a - b).cwiseAbs().maxCoeff();
}
} // namespace

TEST_CASE("Kitti360Calib loads the synthetic calibration", "[calib]")
{
    const fs::path root = MakeCalibTree(MakeTempRoot("calib_load"));
    const Kitti360Calib calib = LoadFixture(root);

    REQUIRE(calib.T_I_Ck.count("image_00") == 1U);
    REQUIRE(calib.T_I_Ck.count("image_01") == 1U);
    REQUIRE(calib.R_rect.count("image_00") == 1U);
    REQUIRE(calib.P_rect.count("image_00") == 1U);
    REQUIRE(calib.K_rect.count("image_00") == 1U);

    CHECK(calib.K_rect.at("image_00").rows() == 3);
    CHECK(calib.K_rect.at("image_00").cols() == 3);
    CHECK(calib.K_rect.at("image_00").isApprox(calib.P_rect.at("image_00").leftCols<3>(), 0.0));

    // K_rect must not be the 3x4 matrix and must not carry the 4th column.
    CHECK(calib.K_rect.at("image_01").isApprox(calib.P_rect.at("image_01").leftCols<3>(), 0.0));

    CHECK(TranslationOf(calib.T_V_C0).isApprox(Eigen::Vector3d(0.10, 0.0, -0.30), 1e-12));
}

TEST_CASE("Camera 1 composition uses its own pose", "[calib]")
{
    const fs::path root = MakeCalibTree(MakeTempRoot("calib_cam1"));
    const Kitti360Calib calib = LoadFixture(root);

    const Eigen::Matrix4d T_W_I = Eigen::Matrix4d::Identity();
    const Eigen::Matrix4d cam0 = ComposeTCkRectV(calib, T_W_I, "image_00");
    const Eigen::Matrix4d cam1 = ComposeTCkRectV(calib, T_W_I, "image_01");
    CHECK(MaxAbsDiff(cam0, cam1) > 0.01);

    const Eigen::Matrix4d T_C0_C1 = Compose(InvertRigid(calib.T_I_Ck.at("image_00")), calib.T_I_Ck.at("image_01"));
    const Eigen::Matrix4d expected =
        Compose(Rect4(calib.R_rect.at("image_01")), InvertRigid(Compose(calib.T_V_C0, T_C0_C1)));
    // R_rect is identity in the fixture, so the manual chain matches.
    CHECK(MaxAbsDiff(cam1, expected) <= 1e-12);
}

TEST_CASE("ComposeTWCkRect matches cam0_to_world and fails on perturbation", "[calib]")
{
    const fs::path root = MakeCalibTree(MakeTempRoot("calib_cam0"));
    const Kitti360Calib calib = LoadFixture(root);
    const auto poses = LoadPoses(root / "data_poses" / kSequence / "poses.txt");
    const auto cam0 = LoadCam0ToWorld(root / "data_poses" / kSequence / "cam0_to_world.txt");
    REQUIRE(!poses.empty());

    for (const auto &[frame, T_W_I] : poses)
    {
        const Eigen::Matrix4d composed = ComposeTWCkRect(calib, T_W_I, "image_00");
        CHECK(MaxAbsDiff(composed, cam0.at(frame)) <= 1e-12);
    }

    // Perturb the stored translation by 1e-2 and require the assertion to fail.
    std::string text;
    for (const int frame : {0, 1, 2})
    {
        text += std::to_string(frame);
        text += " 1 0 0 " + std::to_string(static_cast<double>(frame) + 0.01) + " 0 1 0 0 0 0 1 0 0 0 0 1\n";
    }
    WriteTextFile(root / "data_poses" / kSequence / "cam0_to_world.txt", text);
    const auto perturbed = LoadCam0ToWorld(root / "data_poses" / kSequence / "cam0_to_world.txt");
    double worst = 0.0;
    for (const auto &[frame, T_W_I] : poses)
    {
        worst = std::max(worst, MaxAbsDiff(ComposeTWCkRect(calib, T_W_I, "image_00"), perturbed.at(frame)));
    }
    CHECK(worst > 1e-3);
}

TEST_CASE("ComposeTWCkRect composes the full world chain", "[calib]")
{
    const fs::path root = MakeCalibTree(MakeTempRoot("calib_chain"));
    const Kitti360Calib calib = LoadFixture(root);
    const Eigen::Matrix4d T_W_I = Eigen::Matrix4d::Identity();
    const Eigen::Matrix4d T_W_V = ComposeTWV(calib, T_W_I);
    const Eigen::Matrix4d T_W_C0_rect = ComposeTWCkRect(calib, T_W_I, "image_00");
    const Eigen::Matrix4d T_C0_rect_V = ComposeTCkRectV(calib, T_W_I, "image_00");

    CHECK(MaxAbsDiff(Compose(T_W_C0_rect, T_C0_rect_V), T_W_V) <= 1e-12);
}

TEST_CASE("LoadKitti360Calib rejects a non-orthonormal R_rect", "[calib]")
{
    CalibTreeOptions options;
    options.rectPerturbation = 0.01;
    const fs::path root = MakeCalibTree(MakeTempRoot("calib_bad_rect"), options);
    try
    {
        const Kitti360Calib calib = LoadFixture(root);
        (void)calib;
        FAIL("expected LoadKitti360Calib to reject the perturbed R_rect");
    }
    catch (const std::exception &error)
    {
        const std::string message = error.what();
        CHECK(message.find("R_rect") != std::string::npos);
        CHECK(message.find("orthonormal") != std::string::npos);
    }
}

TEST_CASE("FindCommonFrames returns one contiguous local window", "[calib]")
{
    SECTION("first run wins")
    {
        CalibTreeOptions options;
        options.frames = {0, 1, 2, 7};
        const fs::path root = MakeCalibTree(MakeTempRoot("calib_window_a"), options);
        const auto poses = LoadPoses(root / "data_poses" / kSequence / "poses.txt");
        const auto cam0 = LoadCam0ToWorld(root / "data_poses" / kSequence / "cam0_to_world.txt");
        const std::vector<int> window = FindCommonFrames(root, kSequence, poses, cam0, kCameras, 3);
        CHECK(window == std::vector<int>{0, 1, 2});
    }

    SECTION("skips the isolated head frame")
    {
        CalibTreeOptions options;
        options.frames = {0, 2, 3, 4};
        const fs::path root = MakeCalibTree(MakeTempRoot("calib_window_b"), options);
        const auto poses = LoadPoses(root / "data_poses" / kSequence / "poses.txt");
        const auto cam0 = LoadCam0ToWorld(root / "data_poses" / kSequence / "cam0_to_world.txt");
        const std::vector<int> window = FindCommonFrames(root, kSequence, poses, cam0, kCameras, 3);
        CHECK(window == std::vector<int>{2, 3, 4});
    }

    SECTION("excludes frames missing a component")
    {
        for (const char *component : {"poses", "cam0", "lidar", "images"})
        {
            CalibTreeOptions options;
            options.frames = {0, 1, 2};
            const int omitted = 1;
            const std::string name(component);
            if (name == "poses")
            {
                options.omitPoses.insert(omitted);
            }
            else if (name == "cam0")
            {
                options.omitCam0.insert(omitted);
            }
            else if (name == "lidar")
            {
                options.omitLidar.insert(omitted);
            }
            else
            {
                options.omitImages.insert(omitted);
            }
            const fs::path root = MakeCalibTree(MakeTempRoot("calib_missing_" + name), options);
            const auto poses = LoadPoses(root / "data_poses" / kSequence / "poses.txt");
            const auto cam0 = LoadCam0ToWorld(root / "data_poses" / kSequence / "cam0_to_world.txt");
            const std::vector<int> window = FindCommonFrames(root, kSequence, poses, cam0, kCameras, 1);
            REQUIRE(!window.empty());
            CHECK(std::find(window.begin(), window.end(), omitted) == window.end());
        }
    }
}

TEST_CASE("Pose and calibration loaders reject malformed input", "[calib]")
{
    const fs::path root = MakeTempRoot("calib_badinput");
    const fs::path poses = root / "poses.txt";

    WriteTextFile(poses, "1 1 0 0 0 0 1 0 0 0 0 1 0\n1 1 0 0 0 0 1 0 0 0 0 1 0\n");
    CHECK_THROWS_AS(LoadPoses(poses), IoError);

    WriteTextFile(poses, "1 1 0 0 0 0 1 0 0\n");
    CHECK_THROWS_AS(LoadPoses(poses), IoError);

    WriteTextFile(poses, "1 1 0 0 nan 0 1 0 0 0 0 1 0\n");
    CHECK_THROWS_AS(LoadPoses(poses), IoError);

    const fs::path calibration = root / "calibration";
    fs::create_directories(calibration);
    WriteTextFile(calibration / "calib_cam_to_pose.txt", "image_00: 1 0 0 0 0 1 0 0 0 0 1\n");
    CHECK_THROWS_AS(LoadKitti360Calib(calibration), IoError);
}

TEST_CASE("ComposeTCkRectV agrees across both composition paths", "[calib]")
{
    const fs::path root = MakeCalibTree(MakeTempRoot("calib_paths"));
    const Kitti360Calib calib = LoadFixture(root);
    const auto poses = LoadPoses(root / "data_poses" / kSequence / "poses.txt");
    const Eigen::Matrix4d T_W_I = LookupPose(poses, 1, 5).T_W_I;

    const Eigen::Matrix4d T_C0_C1 = Compose(InvertRigid(calib.T_I_Ck.at("image_00")), calib.T_I_Ck.at("image_01"));
    const Eigen::Matrix4d T_V_Ck = Compose(calib.T_V_C0, T_C0_C1);
    const Eigen::Matrix4d from_raw = Compose(Rect4(calib.R_rect.at("image_01")), InvertRigid(T_V_Ck));
    const Eigen::Matrix4d from_chain = ComposeTCkRectV(calib, T_W_I, "image_01");
    CHECK(MaxAbsDiff(from_chain, from_raw) <= 1e-9);
}
