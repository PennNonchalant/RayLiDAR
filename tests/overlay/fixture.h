#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace raylidar::test
{
/// Intrinsic matrix matching `P_rect.leftCols<3>()` with zero skew.
Eigen::Matrix3d OverlayIntrinsics(double fx, double fy, double cx, double cy);

/// 4x4 translation-only homogeneous transform.
Eigen::Matrix4d Translation4(double x, double y, double z);

/// Row-major RGB buffer filled with a constant value.
std::vector<std::uint8_t> RgbImage(int width, int height, std::uint8_t value);

/// Row-major RGB buffer: white with an axis-aligned black rectangle.
std::vector<std::uint8_t> RgbImageWithBlackRect(int width, int height, int x0, int y0, int x1,
                                                int y1);

/// Row-major 8-bit grayscale buffer: white with an axis-aligned black rectangle.
std::vector<std::uint8_t> GrayImageWithBlackRect(int width, int height, int x0, int y0, int x1,
                                                 int y1);

/// Write a row-major RGB buffer as a PNG (stb is compiled once in the library).
void WriteRgbPng(const std::filesystem::path &path, int width, int height,
                 const std::vector<std::uint8_t> &rgb);

/// Write a PNG signature + IHDR declaring `width` x `height` without pixel data.
/// `stbi_info` can read it, which lets tests exercise the decode size guard.
void WritePngWithDimensions(const std::filesystem::path &path, int width, int height);

/// Read a PNG as a row-major RGB buffer; sets the decoded dimensions.
std::vector<std::uint8_t> ReadRgbPng(const std::filesystem::path &path, int &width, int &height);

/// Fresh directory under the system temp directory.
std::filesystem::path MakeOverlayTempRoot(std::string_view name);

/// Interleaved (x, y, z, intensity) floats for a deterministic synthetic scan.
std::vector<float> OverlayScanPoints(int count);

/// Options controlling MakeOverlayTree.
struct OverlayTreeOptions
{
    int frame = 0;
    int imageWidth = 128;
    int imageHeight = 64;
    int pointCount = 200;
    /// Route `data_rect/<frame>.png` through a symlink to `data_rgb/<frame>.png`.
    bool linkRectToRaw = false;
    bool omitLidar = false;
    bool malformedLidar = false;
    bool omitImage = false;
};

/// Build a synthetic KITTI-360 tree (calibration, poses, LiDAR, PNG images).
std::filesystem::path MakeOverlayTree(const std::filesystem::path &tmp,
                                      const OverlayTreeOptions &options = {});

/// Write a `frames.yaml` pointing at `root`.
std::filesystem::path WriteOverlayConfig(const std::filesystem::path &path,
                                         const std::filesystem::path &root,
                                         const std::string &sequence);
} // namespace raylidar::test
