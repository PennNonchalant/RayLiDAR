#pragma once

#include "rt/GaussianScene.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <span>
#include <vector>

namespace raylidar::simulation
{
struct MeshData
{
    std::vector<std::array<double, 3>> Positions;
    std::vector<std::uint32_t> Indices;
};

class PlyIO
{
  public:
    [[nodiscard]]
    static MeshData LoadMesh(const std::filesystem::path &path);

    [[nodiscard]]
    static std::vector<rt::Gaussian> LoadGaussians(const std::filesystem::path &path,
                                                   const std::array<double, 3> &origin);

    static void WritePointCloud(const std::filesystem::path &path, std::span<const std::array<float, 3>> points);
};
} // namespace raylidar::simulation
