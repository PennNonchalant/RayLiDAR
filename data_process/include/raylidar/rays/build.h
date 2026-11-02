#pragma once

#include <raylidar/export.h>

#include <raylidar/gt/cache.h>
#include <raylidar/rays/rays.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <map>
#include <string>
#include <vector>

namespace raylidar::rays
{
/// Build one camera-origin ray per valid GT point (D-13/D-14, RAY-01/RAY-03).
///
/// `T_Ck_rect_V` must already hold `image_00`/`image_01` transforms composed
/// once per frame with `ComposeTCkRectV`; this function never composes
/// calibration and never recomposes per ray (W11). Points with
/// `camera == 255` or `valid != 1` are skipped; every other point emits exactly
/// one ray (no subsampling). The camera-frame range must lie in `[1.5, 60]` and
/// must agree with the cache's `r_gt` within `1e-3` (float32 tolerance);
/// violations throw `std::invalid_argument`.
RAYLIDAR_EXPORT std::vector<Ray> BuildRays(const std::vector<gt::GtPoint> &gtPoints, int frame,
                                           const std::map<std::string, Eigen::Matrix4d> &T_Ck_rect_V);
} // namespace raylidar::rays
