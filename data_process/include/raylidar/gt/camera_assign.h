#pragma once

#include <raylidar/export.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <cstdint>
#include <map>
#include <string>
#include <utility>

namespace raylidar::gt
{
/// One camera's independent eligibility verdict for a single GT point.
///
/// The three booleans are kept separate on purpose: the GT waterfall must
/// distinguish `no_camera_fov` (`zPositive` or `inImageMargin` false in every
/// camera) from `range_out_of_domain` (the point projects into an image but its
/// camera-frame range lies outside `[rMinM, rMaxM]`).
struct RAYLIDAR_EXPORT CameraEligibility
{
    bool zPositive = false;     ///< The point is in front of the camera.
    bool inImageMargin = false; ///< The pixel lies at least `marginPx` inside the image.
    bool rangeOk = false;       ///< `rMinM <= ||p_C|| <= rMaxM`.
    double u = 0.0;             ///< Pixel column (valid whenever `z > 0`).
    double v = 0.0;             ///< Pixel row (valid whenever `z > 0`).
    double z = 0.0;             ///< Camera-frame depth.
    double rangeM = 0.0;        ///< Camera-frame Euclidean range `||p_C||`.

    [[nodiscard]] bool valid() const
    {
        return zPositive && inImageMargin && rangeOk;
    }
};

/// Evaluate one camera without requiring the point to be eligible.
///
/// The projection is computed with `minRangeM = 0`, `maxRangeM = 1e9`,
/// `marginPx = 0`, and `clampRange = false`, so `u`, `v`, `z`, and `rangeM` are
/// available for every finite point in front of the camera. The three
/// eligibility flags are then derived independently.
RAYLIDAR_EXPORT CameraEligibility EvaluateCameraEligibility(const Eigen::Vector3d &p_L,
                                                            const Eigen::Matrix4d &T_Ck_rect_V,
                                                            const Eigen::Matrix3d &K_rect, int imageWidth,
                                                            int imageHeight, double rMinM = 1.5, double rMaxM = 60.0,
                                                            int marginPx = 5);

/// The frozen primary-camera decision for one GT point.
struct RAYLIDAR_EXPORT CameraAssignment
{
    std::uint8_t camera = 255;                     ///< 0 for `image_00`, 1 for `image_01`, 255 when none.
    double r_gt = -1.0;                            ///< Camera-frame range `||p_C||`; `-1` when unassigned.
    double u = 0.0;                                ///< Pixel column in the assigned camera.
    double v = 0.0;                                ///< Pixel row in the assigned camera.
    Eigen::Vector3d p_C = Eigen::Vector3d::Zero(); ///< Point in the assigned camera frame.
};

/// Deterministically assign one point to at most one camera.
///
/// Cameras are visited in the fixed order `image_00`, `image_01`. A camera is a
/// candidate iff `EvaluateCameraEligibility(...).valid()`. The winner minimizes
/// the normalized radial offset
/// `pow((u - cx) / fx, 2) + pow((v - cy) / fy, 2)`; exact ties keep the lower
/// camera id because only a strictly smaller score replaces the current winner.
/// When no camera is eligible the result is `{camera=255, r_gt=-1}`.
///
/// `T_Ck_rect_V` must be precomputed once per frame (W11): this function never
/// composes calibration transforms and never receives a `Kitti360Calib`.
RAYLIDAR_EXPORT CameraAssignment AssignPrimaryCamera(const Eigen::Vector3d &p_L,
                                                     const std::map<std::string, Eigen::Matrix4d> &T_Ck_rect_V,
                                                     const std::map<std::string, Eigen::Matrix3d> &K_rect,
                                                     const std::map<std::string, std::pair<int, int>> &imageSizes,
                                                     double rMinM = 1.5, double rMaxM = 60.0, int marginPx = 5);
} // namespace raylidar::gt
