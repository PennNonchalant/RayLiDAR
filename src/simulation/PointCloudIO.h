#pragma once

#include <array>
#include <filesystem>
#include <span>

namespace raylidar::simulation
{
// Writes a binary little-endian PLY with float x, y, z per vertex.
void WritePointCloudPly(const std::filesystem::path &path, std::span<const std::array<float, 3>> points);
} // namespace raylidar::simulation
