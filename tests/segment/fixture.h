#pragma once

#include <raylidar/lidar/velodyne.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace raylidar::test
{
/// A synthetic tree with everything `segment-select` and `ScoreFrame` need:
/// calibration, poses, timestamps, Velodyne scans, semantic PNGs, a bbox XML,
/// a phase-2 config, and a frame index.
///
/// The four points per frame exercise the ego/dynamic arithmetic:
///   0: ( 0.1, 0.1,  0.0) — inside the ego ROI AND inside a dynamic bbox
///   1: ( 0.0, 0.0, 10.0) — inside the second dynamic bbox
///   2: ( 5.0, 0.0, 10.0) — static
///   3: (-5.0, 0.0, 10.0) — static
/// Ego removal therefore leaves one dynamic point and two static points.
struct SyntheticSegmentTree
{
    std::filesystem::path root;
    std::filesystem::path datasetRoot;
    std::filesystem::path lidarRoot;
    std::filesystem::path bboxPath;
    std::filesystem::path phaseDir;
    std::filesystem::path frameIndexPath;
    std::filesystem::path configPath;
    std::string sequence;
    std::vector<int> frames;
    std::vector<lidar::PointXYZI> points;
};

/// Fresh directory under the system temp directory.
std::filesystem::path MakeSegmentTempRoot(std::string_view name);

/// Write raw bytes at `path`, creating parent directories as needed.
void WriteSegmentBytes(const std::filesystem::path &path,
                       const std::vector<std::uint8_t> &bytes);

/// Write a UTF-8 text file, creating parent directories as needed.
void WriteSegmentText(const std::filesystem::path &path, std::string_view text);

/// Build the synthetic tree. `usableFrames` is written as the frame index.
SyntheticSegmentTree MakeSyntheticSegmentTree(std::string_view name);

/// Write `<phase_dir>/streams/frame_index.json` with the given usable frames.
void WriteSyntheticFrameIndex(const std::filesystem::path &path,
                              const std::vector<int> &usableFrames,
                              std::string_view sequence);
} // namespace raylidar::test
