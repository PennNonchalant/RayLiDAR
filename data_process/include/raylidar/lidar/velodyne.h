#pragma once

#include <raylidar/export.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string_view>
#include <vector>

namespace raylidar::lidar
{
/// One Velodyne return, exactly as stored in the Nx4 little-endian `.bin`.
/// Field order and point order are part of the data contract: the raw index is
/// the only key shared by the motion mask, the GT cache, and the waterfall.
struct PointXYZI
{
    float x = 0.0F;
    float y = 0.0F;
    float z = 0.0F;
    float intensity = 0.0F;
};

/// A loaded scan. `frame` is informational only; it is never used to reorder or
/// filter the points.
struct VelodyneScan
{
    std::vector<PointXYZI> points;
    int frame = 0;
};

/// Read a binary `float32` little-endian Nx4 scan from `binPath`.
///
/// Throws `raylidar::IoError` (exit 4) when the file is unreadable, empty, not a
/// positive multiple of 16 bytes, or holds more than `maxPoints` returns. The
/// input order is preserved exactly.
RAYLIDAR_EXPORT std::vector<PointXYZI> LoadVelodyneScan(const std::filesystem::path &binPath,
                                                        std::size_t maxPoints = 300000);

/// Strictly parse `YYYY-MM-DD HH:MM:SS.nnnnnnnnn` into nanoseconds since the
/// Unix epoch (UTC). Rejects bad width, non-digits, out-of-range calendar
/// fields, missing or extra fractional digits, and trailing garbage with
/// `raylidar::IoError` (exit 4).
RAYLIDAR_EXPORT std::int64_t ParseUtcTimestamp(std::string_view text);

/// Load one timestamp per non-empty line of `timestamps.txt`.
///
/// Surrounding whitespace is trimmed before parsing; an empty file or any
/// malformed line throws `raylidar::IoError` (exit 4). Line index equals the
/// frame index used by the Velodyne data directory.
RAYLIDAR_EXPORT std::vector<std::int64_t> LoadVelodyneTimestamps(
    const std::filesystem::path &timestampsPath);
} // namespace raylidar::lidar
