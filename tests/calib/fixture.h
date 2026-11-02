#pragma once

#include <Eigen/Core>

#include <filesystem>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace raylidar::test
{
/// Options controlling MakeCalibTree.
struct CalibTreeOptions
{
    /// First-order perturbation added to R_rect_00 in `perspective.txt`
    /// (radians-equivalent). The value 0.01 breaks orthonormality by ~1e-4.
    double rectPerturbation = 0.0;
    /// Translation of `image_01` relative to `image_00` in calib_cam_to_pose.txt.
    Eigen::Vector3d cam1Offset = Eigen::Vector3d(0.0, 0.06, 0.0);
    /// Pose frames to write (removes them from poses.txt when omitted elsewhere).
    std::vector<int> frames = {0, 1, 2};
    /// Emit sloped ground plus wall outliers instead of a flat ground plane.
    bool slopedGround = false;
    /// Frames omitted from poses.txt / cam0_to_world.txt / the LiDAR scan.
    std::set<int> omitPoses;
    std::set<int> omitCam0;
    std::set<int> omitLidar;
    /// Frames that get no rectified image files.
    std::set<int> omitImages;
};

/// Create a fresh directory under the system temp directory and return it.
std::filesystem::path MakeTempRoot(std::string_view name);

/// Build a synthetic KITTI-360 calibration/pose/LiDAR/image tree rooted at `tmp`
/// and return the root. The sequence is `2013_05_28_drive_0000_sync`.
std::filesystem::path MakeCalibTree(const std::filesystem::path &tmp, const CalibTreeOptions &options = {});

/// Write a `frames.yaml` that points at `root`; returns its path.
std::filesystem::path WriteFramesConfig(const std::filesystem::path &path, const std::filesystem::path &root,
                                        const std::string &sequence,
                                        const std::vector<std::string> &cameras = {"image_00", "image_01"});

/// Read an entire text file (used by tests to assert on fixture state).
std::string ReadFixtureFile(const std::filesystem::path &path);
} // namespace raylidar::test
