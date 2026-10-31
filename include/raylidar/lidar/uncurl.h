#pragma once

#include <raylidar/export.h>
#include <raylidar/lidar/velodyne.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace raylidar::lidar
{
/// Rigid ego-motion between two consecutive sweep reference frames.
///
/// `T_W_V_curr` / `T_W_V_next` are `p_W = T_W_V * p_V` at the current and next
/// timestamp. The interval `[t_curr_ns, t_next_ns)` must be strictly positive
/// for `UncurlScan`.
struct UncurlParams
{
    Eigen::Matrix4d T_W_V_curr = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d T_W_V_next = Eigen::Matrix4d::Identity();
    std::int64_t t_curr_ns = 0;
    std::int64_t t_next_ns = 0;
};

/// Per-point time fraction inside the current-to-next sweep, using the official
/// KITTI-360 `curlVelodyneData` convention:
/// `u = 0.5 + atan2(y, x) / (2*pi)`, clamped to `[0, 1)`.
/// A point exactly on `+pi` is clamped to `1 - 1e-12`. Non-finite points throw
/// `std::invalid_argument`.
RAYLIDAR_EXPORT double AzimuthFraction(const Eigen::Vector3d &p);

/// SE(3) interpolation between two rigid transforms.
///
/// The relative transform `T_a^-1 * T_b` is decomposed into a rotation vector
/// and a translation; both are scaled by `u`, then recombined and left-multiplied
/// by `T_a`. Either non-finite input throws `std::invalid_argument`.
RAYLIDAR_EXPORT Eigen::Matrix4d InterpolateRigid(const Eigen::Matrix4d &T_a,
                                                 const Eigen::Matrix4d &T_b, double u);

/// Express a point captured at its azimuth time in the current sensor frame.
///
/// `T_u = InterpolateRigid(T_W_V_curr, T_W_V_next, u)` and
/// `p_ref = inv(T_W_V_curr) * T_u * p_curr`. Non-finite points or transforms
/// throw `std::invalid_argument`; they are never transformed silently.
RAYLIDAR_EXPORT Eigen::Vector3d UncurlPoint(const Eigen::Vector3d &p_curr,
                                            const UncurlParams &params);

/// Uncurl every finite point of a scan, preserving order.
///
/// Non-finite points are skipped and counted in `rejectedNonFinite` (when
/// supplied); intensity is not part of the returned vectors and must be copied
/// unchanged by the caller. Throws `raylidar::IoError` (exit 4) when the sweep
/// interval is not strictly positive.
RAYLIDAR_EXPORT std::vector<Eigen::Vector3d> UncurlScan(
    const std::vector<PointXYZI> &points, const UncurlParams &params,
    std::size_t *rejectedNonFinite = nullptr);
} // namespace raylidar::lidar
