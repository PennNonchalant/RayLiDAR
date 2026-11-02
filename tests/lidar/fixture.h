#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace raylidar::test
{
/// A synthetic 2-frame uncurl fixture: 72 azimuth bins x 120 radial returns of
/// a static world scene plus one dynamic world point, constant 1 m/s motion,
/// 1 s sweeps, calibration, poses, and (optionally) per-point masks.
struct SyntheticUncurlTree
{
    std::filesystem::path root;
    std::filesystem::path datasetRoot;
    std::filesystem::path lidarRoot;
    std::filesystem::path maskDir;
    std::filesystem::path outDir;
    std::filesystem::path configPath;
    std::vector<int> frames;
    std::size_t staticPointCount = 0;
    std::size_t totalPointCount = 0;
};

/// Build the synthetic fixture. When `writeMasks` is true the last point of
/// every frame is marked dynamic; otherwise no mask directory is produced.
SyntheticUncurlTree MakeSyntheticUncurlTree(std::string_view name, bool writeMasks);

/// Write one mask file. `dynamicIndex < 0` marks every point kept static.
void WriteSyntheticMask(const std::filesystem::path &maskDir, int frame, std::size_t pointCount, int dynamicIndex);

/// Create a fresh directory under the system temp directory and return it.
std::filesystem::path MakeLidarTempRoot(std::string_view name);

/// Write a UTF-8 text file, creating parent directories as needed.
void WriteTextFile(const std::filesystem::path &path, std::string_view text);

/// Write one Nx4 little-endian float32 Velodyne `.bin` file.
std::filesystem::path WriteVelodyneBin(const std::filesystem::path &path,
                                       const std::vector<std::array<float, 4>> &points);

/// Write raw bytes at `path`, creating parent directories as needed.
void WriteRawBytes(const std::filesystem::path &path, const std::vector<std::uint8_t> &bytes);
} // namespace raylidar::test
