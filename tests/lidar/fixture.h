#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace raylidar::test
{
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
