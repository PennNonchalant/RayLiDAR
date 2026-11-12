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
struct MeshData;
}

struct raylidar::simulation::MeshData
{
    std::vector<std::array<double, 3>> Positions;
    std::vector<std::uint32_t> Indices;
};

namespace raylidar::simulation
{
class PlyIO;
}

class raylidar::simulation::PlyIO
{
  public:
    [[nodiscard]]
    static MeshData LoadMesh(const std::filesystem::path &path);

    static void WritePointCloud(const std::filesystem::path &path, std::span<const std::array<float, 3>> points);

  private:
    struct PlyHeader;
    class Cursor;

    static PlyHeader ReadPlyHeader(std::istream &file, const std::filesystem::path &path);
};