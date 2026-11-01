#pragma once

#include <raylidar/export.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string_view>
#include <vector>

namespace raylidar::overlay
{
/// Per-point result of projecting a LiDAR scan into a rectified camera image.
/// Every array is in input order and has the same length as the input rows.
struct RAYLIDAR_EXPORT ProjectionResult
{
    std::vector<Eigen::Vector2d> uv;      ///< Pixel coordinates (u, v) in the 1x image.
    std::vector<double> z;                ///< Depth in the camera frame.
    std::vector<double> rangeM;           ///< Euclidean norm of the point in the camera frame.
    std::vector<bool> valid;              ///< Passes z, range, finiteness, and margin checks.
    std::vector<std::size_t> sourceIndex; ///< 0..N-1 input row index.
    int imageWidth = 0;
    int imageHeight = 0;
};

/// Project LiDAR points (in the `V` frame) into a rectified camera image.
///
/// `pointsV` must be `(N, 3)` or `(N, 4)`; only the first three columns are used.
/// `T_Ck_rect_V` maps `V` to the rectified camera frame, and `K` is
/// `P_rect.leftCols<3>()` (no skew term is applied). A point is valid iff every
/// coordinate is finite, `z > 0`, its Euclidean range in the **camera** frame
/// `||p_C||` lies inside `[minRangeM, maxRangeM]` (unless `clampRange` is
/// false), and its pixel lies within `marginPx` of the image border.
RAYLIDAR_EXPORT ProjectionResult ProjectPoints(const Eigen::MatrixXd &pointsV,
                                               const Eigen::Matrix4d &T_Ck_rect_V,
                                               const Eigen::Matrix3d &K, int imageWidth,
                                               int imageHeight, double minRangeM = 10.0,
                                               double maxRangeM = 30.0, int marginPx = 5,
                                               bool clampRange = true);

/// Render projected points over an RGB buffer. Draws a light grid every
/// `gridPx` pixels and a square of half-size `radius` per valid point, coloured
/// by camera-frame range (10 m cyan -> 30 m magenta). When `scale > 1` the
/// result is upscaled with nearest-neighbour sampling. Returns a row-major
/// 3-channel buffer.
RAYLIDAR_EXPORT std::vector<std::uint8_t> RenderOverlay(const std::vector<std::uint8_t> &rgb,
                                                        int width, int height,
                                                        const ProjectionResult &result,
                                                        int scale = 1, int radius = 1,
                                                        int gridPx = 80);

/// Advisory edge-distance proxy. Computes the gradient magnitude, keeps the
/// top 10% as edge pixels, takes the exact Euclidean distance transform, and
/// samples it at projected points. Points that border a depth discontinuity may
/// be preferred; when fewer than 50 exist every valid point is used and
/// `fallback_all_points` is set. The returned JSON keys are `median_px`,
/// `p90_px`, `count`, `fraction_within_2px`, and `fallback_all_points`. The score
/// is advisory only and never encodes the `< 2 px` gate.
RAYLIDAR_EXPORT nlohmann::json EdgeDistanceScore(const std::vector<std::uint8_t> &gray, int width,
                                                 int height, const ProjectionResult &result,
                                                 double discontinuityM = 0.5,
                                                 double neighbourPx = 3.0);

/// Decode one `data_rect` image, project a scan, and write the 1x/4x overlay
/// PNGs plus a JSON summary.
///
/// The PNG header is probed with `stbi_info` first and images larger than 50
/// megapixels are rejected with `std::invalid_argument("image too large")`
/// before allocation. Output paths are exactly
/// `<outDir>/frame_<frame:010d>/<cam>_overlay_1x.png` and `_4x.png`. The 4x
/// image is the 1x bounding box (padded 40 px) magnified with
/// nearest-neighbour sampling; `scale_factor` records the magnification actually
/// applied so a 4x measurement can be converted back to original 1x pixels.
RAYLIDAR_EXPORT nlohmann::json OverlayFrame(
    const std::filesystem::path &imagePath, const Eigen::MatrixXd &pointsV,
    const Eigen::Matrix4d &T_Ck_rect_V, const Eigen::Matrix3d &K,
    const std::filesystem::path &outDir, int frame, std::string_view cam, int scale = 4,
    double minRangeM = 10.0, double maxRangeM = 30.0, int marginPx = 5);
} // namespace raylidar::overlay
