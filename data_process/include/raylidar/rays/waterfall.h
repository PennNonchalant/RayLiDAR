#pragma once

#include <raylidar/export.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <nlohmann/json.hpp>

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace raylidar::rays
{
/// Exclusive per-frame waterfall counts (D-19, RAY-05).
///
/// Every raw LiDAR point is counted exactly once in this order:
/// `nonfinite -> ego_roi -> dynamic_union -> no_camera_fov ->
/// range_out_of_domain -> valid`. The counts use `std::int64_t` so a negative or
/// forged source count can be rejected instead of silently wrapping.
struct RAYLIDAR_EXPORT FrameWaterfall
{
    int frame = 0;
    std::int64_t raw = 0;
    std::int64_t nonfinite = 0;
    std::int64_t ego_roi = 0;
    std::int64_t dynamic_union = 0;
    std::int64_t no_camera_fov = 0;
    std::int64_t range_out_of_domain = 0;
    std::int64_t valid = 0;
};

/// The exclusive reason a kept-static point with `camera == 255` was dropped.
enum class CameraDropClass : std::uint8_t
{
    NoCameraFov = 0,
    RangeOutOfDomain = 1,
};

/// Build one frame's exclusive waterfall.
///
/// `maskCounts` is one entry of `masks/mask_manifest.json` (`counts` object).
/// The function enforces both source closures before returning:
/// `raw == nonfinite + ego_roi + dynamic_union + kept_static` and
/// `kept_static == validRays + noCameraFov + rangeOutOfDomain`. A missing,
/// negative, or inconsistent count throws `raylidar::IoError` with exit code 6;
/// it never guesses a missing value.
RAYLIDAR_EXPORT FrameWaterfall BuildFrameWaterfall(int frame, std::uint64_t raw, const nlohmann::json &maskCounts,
                                                   std::uint64_t keptStatic, std::uint64_t validRays,
                                                   std::uint64_t noCameraFov, std::uint64_t rangeOutOfDomain);

/// Check the exact waterfall equation.
///
/// Returns false and fills `reason` with both sides and the difference when
/// `raw != valid + nonfinite + ego_roi + dynamic_union + no_camera_fov +
/// range_out_of_domain` or when any count is negative. There is no tolerance.
RAYLIDAR_EXPORT bool ValidateWaterfall(const FrameWaterfall &waterfall, std::string *reason = nullptr);

/// Aggregate per-frame waterfalls into `rays/waterfalls.json`.
///
/// The result carries a `frames` array (each entry has `exact`) and a `totals`
/// object with `raw`, `valid`, `coverage = valid/raw`, every exclusion reason,
/// and `exact` (true only when every frame validates exactly). `N_raw/N_valid`
/// and every reason are always reported (D-22).
RAYLIDAR_EXPORT nlohmann::json AggregateWaterfalls(const std::vector<FrameWaterfall> &frames);

/// Classify a kept-static GT point whose primary camera is `255`.
///
/// Calls `EvaluateCameraEligibility` for `image_00` and `image_01` and applies
/// the frozen partition rules (W4):
///  * every camera fails `zPositive` or `inImageMargin` -> `NoCameraFov`;
///  * otherwise, every in-FOV camera fails `rangeOk` -> `RangeOutOfDomain`.
/// A point that is eligible in any camera contradicts the frozen assignment and
/// throws `raylidar::IoError` (exit 6). The two classes therefore partition the
/// camera-255 points exactly.
RAYLIDAR_EXPORT CameraDropClass ClassifyCamera255Point(const Eigen::Vector3d &p_L,
                                                       const std::map<std::string, Eigen::Matrix4d> &T_Ck_rect_V,
                                                       const std::map<std::string, Eigen::Matrix3d> &K_rect,
                                                       const std::map<std::string, std::pair<int, int>> &imageSizes,
                                                       double rMinM = 1.5, double rMaxM = 60.0, int marginPx = 5);
} // namespace raylidar::rays
