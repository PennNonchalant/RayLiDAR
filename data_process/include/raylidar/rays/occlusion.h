#pragma once

#include <raylidar/export.h>

#include <raylidar/gt/cache.h>
#include <raylidar/rays/rays.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <cstdint>
#include <vector>

namespace raylidar::rays
{
/// Frozen GT-only occlusion parameters.
///
/// `depthThresholdM` is authoritative; when it is non-positive the threshold
/// falls back to `sigmaLidarM * sigmaMultiple`. The defaults encode the frozen
/// `3 px / 0.02 m / 3x / 0.06 m` decision (D-18/RAY-04).
struct RAYLIDAR_EXPORT OcclusionOptions
{
    int neighborPx = 3;
    double sigmaLidarM = 0.02;
    double sigmaMultiple = 3.0;
    double depthThresholdM = 0.06;
};

/// Compute the GT-only occlusion flag for one rectified camera.
///
/// **Conservative all-visible z-buffer (W5):** every GT point of the frame is
/// projected into `camera` — not only the points whose stored `camera` equals
/// this camera — because a point assigned to the other camera still physically
/// occludes this one. The projection uses
/// `ProjectPoints(..., minRangeM = 0, maxRangeM = 1e9, marginPx = 0,
/// clampRange = false)` and keeps `z > 0` in-image points.
///
/// Implementation approximation: depths are bucketed by integer pixel, each
/// pixel retains at most the **8 nearest** points, and only the current frame is
/// considered. For each ray assigned to `camera`, the `(2*neighborPx+1)^2`
/// neighbourhood is scanned; the ray is flagged when another GT point's depth is
/// smaller by more than the depth threshold. The ray's own GT point is ignored
/// by matching its original `p_L`. Nothing from the reconstruction is used:
/// the only input is `GtPoint`.
///
/// The returned vector is aligned with `rays`; entries for rays assigned to a
/// different camera are always `0`. The caller merges the flags of
/// `image_00` and `image_01` into `Ray::occluded`.
RAYLIDAR_EXPORT std::vector<std::uint8_t> ComputeOcclusionFlags(const std::vector<gt::GtPoint> &gtPoints,
                                                                const std::vector<Ray> &rays, int camera,
                                                                const Eigen::Matrix4d &T_Ck_rect_V,
                                                                const Eigen::Matrix3d &K_rect, int width, int height,
                                                                const OcclusionOptions &options = {});
} // namespace raylidar::rays
