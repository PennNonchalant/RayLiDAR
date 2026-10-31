#pragma once

#include <raylidar/gt/cache.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace raylidar::test
{
/// Fresh directory under the system temp directory.
std::filesystem::path MakeRaysTempRoot(std::string_view name);

/// 4x4 translation-only homogeneous transform.
Eigen::Matrix4d TranslateRays(double x, double y, double z);

/// Identity transforms for `image_00` and `image_01`.
std::map<std::string, Eigen::Matrix4d> MakeRayTransforms();

/// Build one GT cache point with an explicit camera assignment and range.
raylidar::gt::GtPoint MakeGtPoint(float x, float y, float z, std::uint8_t camera,
                                  std::uint16_t semanticId, float r_gt);

/// A 200x200 rectified camera with fx = fy = 100 and the principal point at
/// (100, 100). Used by the occlusion geometry tests.
Eigen::Matrix3d MakeOcclusionIntrinsics();

/// A synthetic Phase-2 tree usable by both `gt-cache` and `rays-build` smoke
/// runs: it wraps `gt-cache`'s fixture and rewrites the config with the full
/// `rays` section that `rays-build` validates.
struct SyntheticRaysTree
{
    std::filesystem::path root;
    std::filesystem::path phaseDir;
    std::filesystem::path configPath;
    std::filesystem::path gtManifestPath;
    std::string sequence;
    std::vector<int> frames;
    std::size_t keptPerFrame = 0;
};

/// Build the synthetic tree and write the full rays config.
SyntheticRaysTree MakeSyntheticRaysTree(std::string_view name);

/// Run the `gt-cache` smoke pipeline so the tree holds per-frame GT files.
int RunSyntheticGtCache(const SyntheticRaysTree &tree);
} // namespace raylidar::test
