#pragma once

#include "rt/GaussianScene.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <span>
#include <vector>

namespace raylidar::simulation
{
inline constexpr float kGaussianPruneOpacityMin = 0.2f;
inline constexpr float kGaussianPruneMaxScale = 1.0f;

struct MeshData
{
    std::vector<std::array<double, 3>> Positions;
    std::vector<std::uint32_t> Indices;
};

struct GaussianLoadStats
{
    std::size_t Input = 0;
    std::size_t Kept = 0;
    std::size_t RemovedOpacity = 0;
    std::size_t RemovedScale = 0;
};

class PlyIO
{
  public:
    [[nodiscard]]
    static MeshData LoadMesh(const std::filesystem::path &path);

    [[nodiscard]]
    static std::vector<rt::Gaussian> LoadGaussians(const std::filesystem::path &path,
                                                   const std::array<double, 3> &origin, bool prune = true,
                                                   GaussianLoadStats *stats = nullptr);

    static void WritePointCloud(const std::filesystem::path &path, std::span<const std::array<float, 3>> points);
};
} // namespace raylidar::simulation
