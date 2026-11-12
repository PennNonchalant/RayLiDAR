#pragma once

#include <array>
#include <span>
#include <vector>

namespace raylidar::simulation
{
[[nodiscard]]
std::array<double, 3> BoundingBoxCenter(std::span<const std::array<double, 3>> positions);

[[nodiscard]]
std::vector<std::array<float, 3>> ToLocalFloat(std::span<const std::array<double, 3>> positions,
                                               const std::array<double, 3> &origin);
}