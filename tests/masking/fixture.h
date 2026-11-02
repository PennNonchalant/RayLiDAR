#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>

#include <array>
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace raylidar::test
{
/// One `<objectN>` block for the synthetic 3D-bbox XML.
struct BBoxSpec
{
    std::string label = "car";
    int semanticId = 26;
    int timestamp = 0;
    int startFrame = 0;
    int endFrame = 100000;
    bool dynamic = true;
    Eigen::Matrix4d T_W_B = Eigen::Matrix4d::Identity();
};

/// A synthetic tree with everything `mask-build` needs: calibration, poses,
/// timestamps, two Velodyne scans, two semantic PNGs per frame, and a bbox XML.
struct SyntheticMaskTree
{
    std::filesystem::path root;
    std::filesystem::path datasetRoot;
    std::filesystem::path lidarRoot;
    std::filesystem::path bboxPath;
    std::filesystem::path configPath;
    std::filesystem::path phaseDir;
    std::string sequence;
    std::vector<int> frames;
    std::size_t pointCount = 0;
};

/// Fresh directory under the system temp directory.
std::filesystem::path MakeMaskingTempRoot(std::string_view name);

/// Write a UTF-8 text file, creating parent directories as needed.
void WriteTextFile(const std::filesystem::path &path, std::string_view text);

/// Write raw bytes at `path`, creating parent directories as needed.
void WriteRawBytes(const std::filesystem::path &path, const std::vector<std::uint8_t> &bytes);

/// Write a one-channel 8-bit PNG from row-major label bytes.
void WriteGrayscalePng(const std::filesystem::path &path, int width, int height,
                       const std::vector<std::uint8_t> &labels);

/// Serialize `<objectN>` blocks into an OpenCV XML document.
std::string MakeBBoxXml(const std::vector<BBoxSpec> &specs);

/// 4x4 homogeneous matrix from a translation and a uniform scale.
Eigen::Matrix4d TranslatedScaled(double x, double y, double z, double scale);

/// Build the synthetic CLI tree. When `writeSemantics` is false the
/// `image_01` semantics of the last frame are omitted to exercise exit 4.
SyntheticMaskTree MakeSyntheticMaskTree(std::string_view name, bool writeSemantics = true);
} // namespace raylidar::test
