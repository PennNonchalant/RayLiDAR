#pragma once

#include <raylidar/export.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <cstddef>
#include <filesystem>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace raylidar::calib
{
/// All KITTI-360 calibration matrices for one sequence, keyed by camera name
/// (`image_00`, `image_01`, ...).
struct RAYLIDAR_EXPORT Kitti360Calib
{
    Eigen::Matrix4d T_V_C0 = Eigen::Matrix4d::Identity();
    std::map<std::string, Eigen::Matrix4d> T_I_Ck;
    std::map<std::string, Eigen::Matrix3d> R_rect;
    std::map<std::string, Eigen::Matrix<double, 3, 4>> P_rect;
    std::map<std::string, Eigen::Matrix3d> K_rect;
};

/// Load `calib_cam_to_pose.txt`, `calib_cam_to_velo.txt`, and `perspective.txt`
/// from `calibrationDir`.
RAYLIDAR_EXPORT Kitti360Calib LoadKitti360Calib(const std::filesystem::path &calibrationDir);

/// Parse `calib_cam_to_pose.txt`: `image_XX:` followed by 12 floats.
RAYLIDAR_EXPORT std::map<std::string, Eigen::Matrix4d> LoadCalibCamToPose(
    const std::filesystem::path &path);

/// Parse `calib_cam_to_velo.txt`: exactly 12 floats on non-comment lines.
RAYLIDAR_EXPORT Eigen::Matrix4d LoadCalibCamToVelo(const std::filesystem::path &path);

/// Parse `perspective.txt` `P_rect_XX` (12 floats) and `R_rect_XX` (9 floats)
/// entries, keyed by camera name (`image_XX`).
RAYLIDAR_EXPORT std::pair<std::map<std::string, Eigen::Matrix<double, 3, 4>>,
                          std::map<std::string, Eigen::Matrix3d>>
LoadPerspective(const std::filesystem::path &path);

/// Parse `poses.txt`: `frame r11 r12 r13 tx r21 ... tz` (13 tokens). Duplicate
/// frames throw.
RAYLIDAR_EXPORT std::map<int, Eigen::Matrix4d> LoadPoses(const std::filesystem::path &path);

/// Parse `cam0_to_world.txt`: `frame` plus 16 floats. Duplicate frames throw.
RAYLIDAR_EXPORT std::map<int, Eigen::Matrix4d> LoadCam0ToWorld(const std::filesystem::path &path);

/// 4x4 homogeneous matrix from a 3x3 rotation.
RAYLIDAR_EXPORT Eigen::Matrix4d Rect4(const Eigen::Matrix3d &R);

/// `T_W_V = T_W_I * T_I_C0 * inv(T_V_C0)`.
RAYLIDAR_EXPORT Eigen::Matrix4d ComposeTWV(const Kitti360Calib &calib,
                                           const Eigen::Matrix4d &T_W_I);

/// `T_W_Ck_rect = T_W_I * T_I_Ck[cam] * inv(R_rect4(cam))`.
RAYLIDAR_EXPORT Eigen::Matrix4d ComposeTWCkRect(const Kitti360Calib &calib,
                                                const Eigen::Matrix4d &T_W_I,
                                                std::string_view cam);

/// `T_Ck_rect_V = inv(T_W_Ck_rect) * T_W_V`. Also asserts that it equals
/// `R_rect4(cam) * inv(T_V_Ck)` within 1e-9 (two independent composition paths).
RAYLIDAR_EXPORT Eigen::Matrix4d ComposeTCkRectV(const Kitti360Calib &calib,
                                                const Eigen::Matrix4d &T_W_I,
                                                std::string_view cam);

/// Return one contiguous local window of up to `requested` frames that exist in
/// both pose maps and have every camera image and the Velodyne scan. The first
/// run of consecutive frame IDs long enough wins; otherwise the longest run is
/// returned. The result is deduplicated and ascending.
RAYLIDAR_EXPORT std::vector<int> FindCommonFrames(
    const std::filesystem::path &root, std::string_view sequence,
    const std::map<int, Eigen::Matrix4d> &poses,
    const std::map<int, Eigen::Matrix4d> &cam0ToWorld,
    const std::vector<std::string> &cameras, std::size_t requested);
} // namespace raylidar::calib
