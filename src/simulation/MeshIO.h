#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace raylidar::simulation
{
// Triangle mesh in double precision, exactly as stored on disk.
struct MeshData
{
    std::vector<std::array<double, 3>> Positions;
    std::vector<std::uint32_t> Indices;
};

// Reads a binary little-endian PLY. Vertex x/y/z may be float or double; polygons are fan-triangulated.
[[nodiscard]]
MeshData LoadPlyMesh(const std::filesystem::path &path);

[[nodiscard]]
std::array<double, 3> BoundingBoxCenter(std::span<const std::array<double, 3>> positions);

// Subtracts origin and converts to float. World coordinates in the thousands of metres only keep ~0.2 mm of
// float precision, so geometry and rays are shifted into a local frame before going to the GPU.
[[nodiscard]]
std::vector<std::array<float, 3>> ToLocalFloat(std::span<const std::array<double, 3>> positions,
                                               const std::array<double, 3> &origin);
} // namespace raylidar::simulation
