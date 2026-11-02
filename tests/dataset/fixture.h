#pragma once

#include <filesystem>
#include <set>
#include <string>
#include <string_view>
#include <utility>

namespace raylidar::test
{
/// Options controlling MakeDatasetTree.
///
/// `missing` tokens are component-level (`calibration`, `poses`, `images`,
/// `semantics`, `lidar`) or specific relative paths
/// (`calibration/calib_cam_to_pose.txt`, `poses/poses.txt`, ...).
struct DatasetTreeOptions
{
    bool official = true;
    bool kittiRawCalib = false;
    bool repackaged = false;
    std::set<std::string> missing;
    std::pair<int, int> imageCounts = {2, 2};
    std::pair<int, int> imageSize = {1408, 376};
};

/// Create a fresh directory under the system temp directory and return it.
std::filesystem::path MakeTempRoot(std::string_view name);

/// Build a synthetic dataset tree rooted at `tmp` and return the root.
std::filesystem::path MakeDatasetTree(const std::filesystem::path &tmp, const DatasetTreeOptions &options);

/// Read an entire text file (used by tests to assert on fixture state).
std::string ReadFixtureFile(const std::filesystem::path &path);
} // namespace raylidar::test
