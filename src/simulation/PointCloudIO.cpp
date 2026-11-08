#include "PointCloudIO.h"

#include <bit>
#include <fstream>
#include <stdexcept>

namespace raylidar::simulation
{
void WritePointCloudPly(const std::filesystem::path &path, std::span<const std::array<float, 3>> points)
{
    static_assert(std::endian::native == std::endian::little, "PLY writer assumes a little-endian host.");

    std::ofstream file(path, std::ios::binary);

    if (!file)
    {
        throw std::runtime_error("Failed to open PLY file for writing: " + path.string());
    }

    file << "ply\n"
         << "format binary_little_endian 1.0\n"
         << "element vertex " << points.size() << '\n'
         << "property float x\n"
         << "property float y\n"
         << "property float z\n"
         << "end_header\n";

    const auto bytes = std::as_bytes(points);
    file.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));

    if (!file)
    {
        throw std::runtime_error("Failed to write PLY file: " + path.string());
    }
}
} // namespace raylidar::simulation
