#pragma once

#include <raylidar/calib/kitti360.h>
#include <raylidar/common/mask_bits.h>
#include <raylidar/export.h>
#include <raylidar/lidar/velodyne.h>
#include <raylidar/masking/bbox.h>
#include <raylidar/masking/semantics.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace raylidar::masking
{
/// Frozen D-08/D-09 mask parameters, read from `configs/phase2.yaml`.
struct RAYLIDAR_EXPORT MaskOptions
{
    double egoRoiXMin = -2.5;
    double egoRoiXMax = 2.5;
    double egoRoiYMin = -1.2;
    double egoRoiYMax = 1.2;
    double egoRoiZMin = -2.0;
    double egoRoiZMax = 0.3;
    int bboxMaxTimeGapFrames = 2;
    double residualDynamicRadiusM = 1.0;
};

/// One per-frame mask artifact before it is written to disk.
///
/// `bits` holds one frozen `MaskBit` byte per raw point, in raw input order.
/// The pixel masks are row-major binary buffers (`0`/`1`) with the dimensions
/// recorded in `imageSizes`.
struct RAYLIDAR_EXPORT MaskResult
{
    std::vector<std::uint8_t> bits;
    std::vector<std::uint8_t> image_00_pixels;
    std::vector<std::uint8_t> image_01_pixels;
    std::map<std::string, std::pair<int, int>> imageSizes;
    nlohmann::json counts;
    nlohmann::json residual;
};

/// Classify every raw point against both independent dynamic sources and the
/// D-09 ego box.
///
/// `uncurledPoints` is index-aligned with `rawPoints` (non-finite raw points
/// carry a non-finite uncurled placeholder). Both dynamic sources are computed
/// independently and kept as separate bits; `kDynamicUnion` is their OR. The
/// ego ROI is evaluated on raw `p_V`; the 2D source is evaluated on uncurled
/// `p_L` with `clampRange=false` so a far dynamic label still counts.
RAYLIDAR_EXPORT MaskResult BuildMask(
    const std::vector<lidar::PointXYZI> &rawPoints,
    const std::vector<Eigen::Vector3d> &uncurledPoints, int frame,
    const calib::Kitti360Calib &calib, const Eigen::Matrix4d &T_W_V,
    const std::vector<DynamicBBox> &boxes,
    const std::map<std::string, SemanticImage> &semantics,
    const std::map<std::string, std::pair<int, int>> &imageSizes,
    const std::map<std::string, Eigen::Matrix4d> &T_Ck_rect_V, const MaskOptions &options);

/// Write the per-frame artifact into `masksDir`.
///
/// Exactly `frame_<frame:010d>/points.bin`, `image_00_dynamic.png`,
/// `image_01_dynamic.png`, and `mask.json` are written. The returned JSON is
/// the content of `mask.json`; it records per-frame SHA-256 hashes, counts,
/// residual dynamic statistics, and the four downstream consumers.
RAYLIDAR_EXPORT nlohmann::json WriteMaskArtifact(
    const std::filesystem::path &masksDir, int frame, const MaskResult &result,
    const std::vector<lidar::PointXYZI> &rawPoints, std::string_view sequence);

/// Aggregate per-frame JSON documents into `mask_manifest.json`.
///
/// When `manifestFileName` is set to a non-final name (e.g. a smoke run) the
/// final manifest is never overwritten. `configSha256` is recorded verbatim and
/// may be empty in tests.
RAYLIDAR_EXPORT nlohmann::json BuildMaskManifest(
    const std::filesystem::path &masksDir, const std::vector<nlohmann::json> &frames,
    std::string_view sequence, std::string_view configSha256 = {},
    std::string_view manifestFileName = "mask_manifest.json");
} // namespace raylidar::masking
