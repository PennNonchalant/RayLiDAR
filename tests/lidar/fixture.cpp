#include "fixture.h"

#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <system_error>

namespace raylidar::test
{
namespace fs = std::filesystem;

fs::path MakeLidarTempRoot(std::string_view name)
{
    std::error_code error;
    const fs::path base = fs::temp_directory_path(error) / "raylidar_lidar_tests";
    fs::create_directories(base, error);
    static int counter = 0;
    const fs::path root = base / (std::string(name) + "_" + std::to_string(++counter));
    fs::remove_all(root, error);
    fs::create_directories(root, error);
    return root;
}

void WriteTextFile(const fs::path &path, std::string_view text)
{
    std::error_code error;
    fs::create_directories(path.parent_path(), error);
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output.good())
    {
        throw std::runtime_error("cannot write " + path.string());
    }
    output.write(text.data(), static_cast<std::streamsize>(text.size()));
}

void WriteRawBytes(const fs::path &path, const std::vector<std::uint8_t> &bytes)
{
    std::error_code error;
    fs::create_directories(path.parent_path(), error);
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output.good())
    {
        throw std::runtime_error("cannot write " + path.string());
    }
    output.write(reinterpret_cast<const char *>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
}

fs::path WriteVelodyneBin(const fs::path &path,
                          const std::vector<std::array<float, 4>> &points)
{
    std::error_code error;
    fs::create_directories(path.parent_path(), error);
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output.good())
    {
        throw std::runtime_error("cannot write " + path.string());
    }
    for (const auto &point : points)
    {
        output.write(reinterpret_cast<const char *>(point.data()),
                     static_cast<std::streamsize>(sizeof(float) * point.size()));
    }
    return path;
}
} // namespace raylidar::test
