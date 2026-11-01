#pragma once

#include <raylidar/export.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <cstdint>
#include <filesystem>
#include <string_view>
#include <vector>

namespace raylidar::gt
{
/// One per-frame static GT point.
///
/// `p_L` is the uncurled point in the LiDAR (`V`) frame, `frame` is the sweep
/// id, `semanticId` is the rectified-image label at the assigned pixel
/// (`65535` when no label was available), `camera` is the frozen primary
/// camera (`0`, `1`, or `255`), `valid` is `1` for every cached point, and
/// `r_gt` is the **camera-frame** range `||p_C||` (`-1` when `camera == 255`).
struct RAYLIDAR_EXPORT GtPoint
{
    Eigen::Vector3f p_L = Eigen::Vector3f::Zero();
    std::uint32_t frame = 0;
    std::uint16_t semanticId = 0;
    std::uint8_t camera = 255;
    std::uint8_t valid = 0;
    float r_gt = -1.0F;
};

/// Write one per-frame GT cache file.
///
/// The header is exactly 64 bytes: magic `"RLGT"` (4), version `u16 = 1`,
/// record size `u16 = 32`, count `u32`, frame `u32`, reserved 48. Each record is
/// exactly 32 bytes: `p_L f32[3]`, `r_gt f32`, `frame u32`, `semanticId u16`,
/// `camera u8`, `valid u8`, `pad[8]`. All integers are little-endian.
///
/// Non-finite coordinates or ranges are rejected with
/// `std::invalid_argument`; they are never written silently.
RAYLIDAR_EXPORT void WriteGtFrame(const std::filesystem::path &path, int frame,
                                  const std::vector<GtPoint> &points);

/// Read one per-frame GT cache file.
///
/// Rejects a wrong magic, wrong version, wrong record size, an exact file
/// length that differs from `64 + count * 32`, a record frame inconsistent with
/// the header, and non-finite coordinates with `raylidar::IoError` (exit 4).
/// When `frame` is non-null it receives the header frame id.
RAYLIDAR_EXPORT std::vector<GtPoint> ReadGtFrame(const std::filesystem::path &path,
                                                 int *frame = nullptr);

} // namespace raylidar::gt
