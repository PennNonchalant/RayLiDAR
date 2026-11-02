#pragma once

#include <raylidar/export.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <map>
#include <utility>
#include <vector>

namespace raylidar::calib
{
/// Compose two rigid transforms. `Compose(T_A_B, T_B_C)` returns `T_A_B * T_B_C`
/// with the naming rule `p_A = T_AB * p_B`. Both inputs and the result must be
/// finite 4x4 matrices.
RAYLIDAR_EXPORT Eigen::Matrix4d Compose(const Eigen::Matrix4d &T_A_B, const Eigen::Matrix4d &T_B_C);

/// Lift a 3x4 or 4x4 matrix to a 4x4 homogeneous transform with a
/// `[0, 0, 0, 1]` bottom row. Other shapes and non-finite values throw.
RAYLIDAR_EXPORT Eigen::Matrix4d AsHomogeneous(const Eigen::MatrixXd &mat);

/// Validate the rotation block and return the analytic inverse
/// `[[R^T, -R^T t], [0, 1]]`. The analytic inverse is compared with the full
/// 4x4 inverse using a translation-scale-aware tolerance
/// `atol + relAtol * max(1, ||t||)`, because official-file rotations carry
/// ~1e-6 noise while `T_W_C0_rect` has `||t|| ~ 4 km`.
RAYLIDAR_EXPORT Eigen::Matrix4d InvertRigid(const Eigen::Matrix4d &T, double atol = 1e-9, double relAtol = 2e-6);

[[nodiscard]] RAYLIDAR_EXPORT Eigen::Matrix3d RotationOf(const Eigen::Matrix4d &T);
[[nodiscard]] RAYLIDAR_EXPORT Eigen::Vector3d TranslationOf(const Eigen::Matrix4d &T);

/// Throw unless `|det(R) - 1| <= detAtol` and `max|R^T R - I| <= atol`.
/// Defaults are calibrated to official-file precision (measured 9.93e-7) and
/// still reject a scaled, reflected, or 1e-3-perturbed rotation.
RAYLIDAR_EXPORT void AssertRotation(const Eigen::Matrix3d &R, double atol = 2e-6, double detAtol = 2e-6);

/// Throw unless `T_AB * T_BA` is the identity within `atol`. Use it on
/// camera-relative transforms or rotation blocks; world-frame translations carry
/// source-precision noise that is larger than this gate.
RAYLIDAR_EXPORT void AssertRoundtrip(const Eigen::Matrix4d &T_AB, const Eigen::Matrix4d &T_BA, double atol = 1e-9);

/// SEG frame (scene-local frame) from camera centres in world coordinates.
/// Returns `{T_S_W, T_W_S}` where the origin of `S` is the component-wise
/// median of `camCentersW` and the rotation is identity.
RAYLIDAR_EXPORT std::pair<Eigen::Matrix4d, Eigen::Matrix4d> MakeSegTransform(
    const std::vector<Eigen::Vector3d> &camCentersW);

/// Result of a sparse pose lookup.
struct RAYLIDAR_EXPORT PoseLookup
{
    int frameRequested = 0;
    int poseFrame = 0;
    int delta = 0; ///< poseFrame - frameRequested
    Eigen::Matrix4d T_W_I = Eigen::Matrix4d::Identity();
};

/// Resolve a frame against a sparse pose map. An exact hit wins. Otherwise the
/// nearest pose frame is used (ties go to the lower frame). A nearest delta
/// greater than `maxDelta` throws `std::out_of_range`; there is never a
/// fallback to the first pose or frame 0.
RAYLIDAR_EXPORT PoseLookup LookupPose(const std::map<int, Eigen::Matrix4d> &poses, int frame, int maxDelta = 5);

/// Ground plane in the `V` (LiDAR) frame: `normal . p + offset = 0`, `||normal|| = 1`.
struct RAYLIDAR_EXPORT GroundPlane
{
    Eigen::Vector3d normal = Eigen::Vector3d::UnitZ();
    double offset = 0.0;
};

/// Robust local ground plane. Keeps returns with horizontal radius in
/// `[radiusMin, radiusMax]`, keeps the single lowest return per azimuth sector,
/// rejects candidates outside `[zMin, zMax]`, and fits `z = a x + b y + c` with
/// iteratively trimmed least squares. Throws `std::invalid_argument` when fewer
/// than `minInliers` candidates survive.
RAYLIDAR_EXPORT GroundPlane EstimateGroundPlane(const Eigen::MatrixXd &pointsV, double radiusMin = 5.0,
                                                double radiusMax = 30.0, double zMin = -3.0, double zMax = 1.0,
                                                int azimuthBins = 240, double trimFraction = 0.7, int minInliers = 100);

/// Perpendicular distance from a camera centre to the ground plane.
RAYLIDAR_EXPORT double CameraHeight(const GroundPlane &plane, const Eigen::Vector3d &cameraCentreV);

/// Convenience overload using `TranslationOf(T_V_C0)` as the camera centre.
RAYLIDAR_EXPORT double CameraHeight(const Eigen::Matrix4d &T_V_C0, const GroundPlane &plane);
} // namespace raylidar::calib
