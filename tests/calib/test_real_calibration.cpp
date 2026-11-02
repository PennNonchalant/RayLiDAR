#include <raylidar/calib/frames.h>
#include <raylidar/calib/kitti360.h>

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace
{
namespace fs = std::filesystem;
using raylidar::calib::AssertRotation;
using raylidar::calib::AssertRoundtrip;
using raylidar::calib::CameraHeight;
using raylidar::calib::Compose;
using raylidar::calib::ComposeTCkRectV;
using raylidar::calib::ComposeTWCkRect;
using raylidar::calib::EstimateGroundPlane;
using raylidar::calib::FindCommonFrames;
using raylidar::calib::GroundPlane;
using raylidar::calib::InvertRigid;
using raylidar::calib::LoadCam0ToWorld;
using raylidar::calib::LoadKitti360Calib;
using raylidar::calib::LoadPoses;
using raylidar::calib::LookupPose;
using raylidar::calib::Rect4;
using raylidar::calib::RotationOf;
using raylidar::calib::TranslationOf;

const fs::path kRoot = "/data-20T/hybrid-lidar-sim/raw/kitti360";
const std::string kSequence = "2013_05_28_drive_0000_sync";
const std::vector<std::string> kCameras = {"image_00", "image_01"};
constexpr double kRoundtripAtol = 1.0e-9;
constexpr double kCam0WorldAtol = 1.0e-3;
constexpr double kHeightMin = 1.4;
constexpr double kHeightMax = 1.9;
constexpr double kHeightStability = 0.1;

bool RealDataPresent()
{
    return fs::is_regular_file(kRoot / "calibration" / "calib_cam_to_pose.txt") &&
           fs::is_regular_file(kRoot / "calibration" / "calib_cam_to_velo.txt") &&
           fs::is_regular_file(kRoot / "calibration" / "perspective.txt") &&
           fs::is_regular_file(kRoot / "data_poses" / kSequence / "poses.txt") &&
           fs::is_regular_file(kRoot / "data_poses" / kSequence / "cam0_to_world.txt");
}

Eigen::MatrixXd ReadLidar(const fs::path &path)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    REQUIRE(input.good());
    const std::streamsize bytes = input.tellg();
    REQUIRE(bytes > 0);
    REQUIRE(bytes % 16 == 0);
    input.seekg(0, std::ios::beg);
    std::vector<float> raw(static_cast<std::size_t>(bytes) / sizeof(float));
    input.read(reinterpret_cast<char *>(raw.data()), bytes);

    const std::size_t count = raw.size() / 4U;
    Eigen::MatrixXd points(static_cast<Eigen::Index>(count), 3);
    for (std::size_t index = 0; index < count; ++index)
    {
        points(static_cast<Eigen::Index>(index), 0) = raw[index * 4U + 0U];
        points(static_cast<Eigen::Index>(index), 1) = raw[index * 4U + 1U];
        points(static_cast<Eigen::Index>(index), 2) = raw[index * 4U + 2U];
    }
    return points;
}

double MaxAbsDiff(const Eigen::Matrix4d &a, const Eigen::Matrix4d &b)
{
    return (a - b).cwiseAbs().maxCoeff();
}

std::string FrameName(int frame)
{
    std::string name = std::to_string(frame);
    return std::string(name.size() < 10U ? 10U - name.size() : 0U, '0') + name;
}

fs::path LidarPath(int frame)
{
    return kRoot / "data_3d_raw" / kSequence / "velodyne_points" / "data" / (FrameName(frame) + ".bin");
}
} // namespace

TEST_CASE("Real KITTI-360 calibration satisfies the G1 tolerances", "[calib]")
{
    if (!RealDataPresent())
    {
        SKIP("real calibration absent");
    }

    const auto calib = LoadKitti360Calib(kRoot / "calibration");
    const auto poses = LoadPoses(kRoot / "data_poses" / kSequence / "poses.txt");
    const auto cam0 = LoadCam0ToWorld(kRoot / "data_poses" / kSequence / "cam0_to_world.txt");
    REQUIRE(!poses.empty());
    REQUIRE(!cam0.empty());

    SECTION("R_rect determinant and orthonormality")
    {
        for (const auto &[camera, rectification] : calib.R_rect)
        {
            INFO("camera " << camera);
            CHECK_NOTHROW(AssertRotation(rectification));
        }
    }

    SECTION("pose rotations are orthonormal")
    {
        std::size_t checked = 0;
        for (const auto &[frame, transform] : poses)
        {
            INFO("frame " << frame);
            CHECK_NOTHROW(AssertRotation(RotationOf(transform)));
            if (++checked >= 100U)
            {
                break;
            }
        }
        CHECK(checked > 0U);
    }

    SECTION("transform chain, inverse and camera height on the sampled window")
    {
        const std::vector<int> window = FindCommonFrames(kRoot, kSequence, poses, cam0, kCameras, 3);
        REQUIRE(window.size() == 3U);

        std::vector<double> heights;
        double maxCam0Delta = 0.0;
        for (const int frame : window)
        {
            INFO("frame " << frame);
            const auto lookup = LookupPose(poses, frame, 5);
            CHECK(lookup.poseFrame == frame);

            const Eigen::Matrix4d T_W_V = ComposeTWV(calib, lookup.T_W_I);
            (void)T_W_V;

            const Eigen::Matrix4d T_W_C0_rect = ComposeTWCkRect(calib, lookup.T_W_I, "image_00");
            REQUIRE(cam0.count(frame) == 1U);
            maxCam0Delta = std::max(maxCam0Delta, MaxAbsDiff(T_W_C0_rect, cam0.at(frame)));

            // Scale-aware analytic inverse (1e-9 + 2e-6 * ||t||), not a fixed 1e-12.
            CHECK_NOTHROW(InvertRigid(T_W_C0_rect));

            // ComposeTCkRectV internally asserts the rectified/raw paths agree at 1e-9.
            const Eigen::Matrix4d T_Ck_rect_V = ComposeTCkRectV(calib, lookup.T_W_I, "image_00");

            // Camera-relative raw round trip at the strict gate.
            const Eigen::Matrix4d T_C0_Ck =
                Compose(InvertRigid(calib.T_I_Ck.at("image_00")), calib.T_I_Ck.at("image_00"));
            const Eigen::Matrix4d T_V_Ck = Compose(calib.T_V_C0, T_C0_Ck);
            const Eigen::Matrix4d T_Ck_V = InvertRigid(T_V_Ck);
            CHECK_NOTHROW(AssertRoundtrip(T_Ck_V, InvertRigid(T_Ck_V), kRoundtripAtol));

            // The rectified camera-relative transform matches R_rect * T_Ck_V at 1e-9.
            const Eigen::Matrix4d from_raw = Compose(Rect4(calib.R_rect.at("image_00")), T_Ck_V);
            CHECK(MaxAbsDiff(T_Ck_rect_V, from_raw) <= kRoundtripAtol);

            const Eigen::MatrixXd pointsV = ReadLidar(LidarPath(frame));
            const GroundPlane plane = EstimateGroundPlane(pointsV);
            heights.push_back(CameraHeight(calib.T_V_C0, plane));
        }

        CHECK(maxCam0Delta <= kCam0WorldAtol);
        CHECK(maxCam0Delta > 0.0);
        REQUIRE(heights.size() == window.size());

        const auto [minimum, maximum] = std::minmax_element(heights.begin(), heights.end());
        for (const double height : heights)
        {
            CHECK(height >= kHeightMin);
            CHECK(height <= kHeightMax);
        }
        CHECK((*maximum - *minimum) <= kHeightStability);
    }
}
