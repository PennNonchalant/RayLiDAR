#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace raylidar::test
{
/// Two rectified cameras with a 400x300 sensor and a 100 px focal length.
/// `k` is the shared intrinsic matrix; tests replace individual transforms to
/// exercise the deterministic normalized-radius assignment.
struct TwoCameraRig
{
    Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
    std::map<std::string, Eigen::Matrix4d> T_Ck_rect_V;
    std::map<std::string, Eigen::Matrix3d> K_rect;
    std::map<std::string, std::pair<int, int>> imageSizes;
};

/// A rig where both cameras share `K` and an identity `T_Ck_rect_V`.
TwoCameraRig MakeTwoCameraRig();

/// 4x4 transform with a translation and an identity rotation.
Eigen::Matrix4d Translate(double x, double y, double z);

/// A synthetic Phase-2 tree with everything `gt-cache` needs: calibration,
/// poses, timestamps, Velodyne scans, semantic PNGs, per-frame mask bit files,
/// a full segment manifest, and a phase-2 config.
///
/// Each frame holds four points; the mask keeps the first two and marks the
/// third dynamic and the fourth ego. The first two project to distinct pixels
/// in both cameras, so the synthetic cache has exactly two GT points per frame.
struct SyntheticGtTree
{
    std::filesystem::path root;
    std::filesystem::path datasetRoot;
    std::filesystem::path lidarRoot;
    std::filesystem::path phaseDir;
    std::filesystem::path configPath;
    std::filesystem::path segmentManifestPath;
    std::filesystem::path posesPath;
    std::filesystem::path timestampsPath;
    std::string sequence;
    std::vector<int> frames;
    std::size_t keptPerFrame = 0;
    std::size_t pointCount = 0;
};

/// Fresh directory under the system temp directory.
std::filesystem::path MakeGtTempRoot(std::string_view name);

/// Write a UTF-8 text file, creating parent directories as needed.
void WriteGtTextFile(const std::filesystem::path &path, std::string_view text);

/// Write raw bytes at `path`, creating parent directories as needed.
void WriteGtRawBytes(const std::filesystem::path &path, const std::vector<std::uint8_t> &bytes);

/// Write a one-channel 8-bit PNG from row-major label bytes.
void WriteGtGrayscalePng(const std::filesystem::path &path, int width, int height,
                         const std::vector<std::uint8_t> &labels);

/// Build the synthetic CLI tree used by the `gt-cache` integration tests.
SyntheticGtTree MakeSyntheticGtTree(std::string_view name);

/// Overwrite `<phase_dir>/segment/segment_manifest.json`.
///
/// `mode` is written verbatim; `frameCount` is the `selected.frame_count` field
/// and `frames` is `selected.frames`. When `mode` is not "full" the frame ids
/// are still written so a caller can exercise the mode gate first.
void WriteSyntheticSegmentManifest(const std::filesystem::path &path, const std::vector<int> &frames, int frameCount,
                                   std::string_view mode = "full");
} // namespace raylidar::test
