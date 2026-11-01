#pragma once

#include <raylidar/export.h>

#include <cstdint>
#include <vector>

namespace raylidar::overlay
{
/// Exact Euclidean distance (in pixels) from every `0` pixel to the nearest
/// non-zero pixel. The input is row-major with exactly `width * height`
/// entries; `0` is the background and any non-zero value is foreground.
///
/// Uses the Felzenszwalb-Huttenlocher 1-D lower-envelope algorithm on squared
/// distances, then takes the square root. Pixels with no foreground in the
/// whole image return `NaN`. Throws `std::invalid_argument` when the dimensions
/// are non-positive or do not match the buffer size.
RAYLIDAR_EXPORT std::vector<double> EuclideanDistanceTransform(
    const std::vector<std::uint8_t> &binary, int width, int height);

/// Gradient magnitude of an 8-bit single-channel image using central
/// differences with clamped borders. Row-major, `width * height` entries.
RAYLIDAR_EXPORT std::vector<double> GradientMagnitude(const std::vector<std::uint8_t> &gray,
                                                      int width, int height);

/// Linear-interpolated percentile of a copy of `values`. Returns `NaN` for an
/// empty input or a non-finite fraction. Fractions are clamped to `[0, 1]`.
RAYLIDAR_EXPORT double Percentile(std::vector<double> values, double fraction);
} // namespace raylidar::overlay
