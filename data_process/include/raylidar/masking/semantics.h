#pragma once

#include <raylidar/export.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace raylidar::masking
{
/// One decoded KITTI-360 semantic label image.
///
/// The PNG is indexed in rectified `data_rect` image space, matching the
/// official viewer pairing, and is stored row-major with one 16-bit label per
/// pixel. 8-bit sources are widened without rescaling.
struct RAYLIDAR_EXPORT SemanticImage
{
    int width = 0;
    int height = 0;
    std::vector<std::uint16_t> labels;
};

/// Load one single-channel semantic PNG.
///
/// `stbi_info` is probed before allocation; images above `maxPixels`, images
/// with more than one channel, and unreadable files throw `raylidar::IoError`
/// (exit 4). 16-bit PNGs are read through `stbi_load_16`; 8-bit PNGs through
/// `stbi_load` and widened to `uint16_t`.
RAYLIDAR_EXPORT SemanticImage LoadSemantics(const std::filesystem::path &pngPath, std::size_t maxPixels = 50000000);

/// True for the frozen dynamic label set `{24, 25, 26, 27, 28, 32, 33}`.
RAYLIDAR_EXPORT bool IsDynamicLabel(std::uint16_t labelId);

/// Nearest-pixel lookup: `floor(u + 0.5)`, `floor(v + 0.5)`. Out-of-bounds or
/// non-finite coordinates return `65535`.
RAYLIDAR_EXPORT std::uint16_t LabelAt(const SemanticImage &image, double u, double v);
} // namespace raylidar::masking
