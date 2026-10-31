#include "fixture.h"

#include "../gt/fixture.h"

#include <raylidar/cli/commands_phase2.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace raylidar::test
{
namespace
{
namespace fs = std::filesystem;
} // namespace

fs::path MakeRaysTempRoot(const std::string_view name)
{
    return MakeGtTempRoot(name);
}

Eigen::Matrix4d TranslateRays(const double x, const double y, const double z)
{
    return Translate(x, y, z);
}

std::map<std::string, Eigen::Matrix4d> MakeRayTransforms()
{
    return {{"image_00", Eigen::Matrix4d::Identity()},
            {"image_01", Eigen::Matrix4d::Identity()}};
}

gt::GtPoint MakeGtPoint(const float x, const float y, const float z, const std::uint8_t camera,
                        const std::uint16_t semanticId, const float r_gt)
{
    gt::GtPoint point;
    point.p_L = Eigen::Vector3f(x, y, z);
    point.frame = 0U;
    point.semanticId = semanticId;
    point.camera = camera;
    point.valid = 1U;
    point.r_gt = r_gt;
    return point;
}

Eigen::Matrix3d MakeOcclusionIntrinsics()
{
    Eigen::Matrix3d intrinsics = Eigen::Matrix3d::Identity();
    intrinsics << 100.0, 0.0, 100.0,
                  0.0, 100.0, 100.0,
                  0.0, 0.0, 1.0;
    return intrinsics;
}

SyntheticRaysTree MakeSyntheticRaysTree(const std::string_view name)
{
    const SyntheticGtTree gtTree = MakeSyntheticGtTree(name);

    SyntheticRaysTree tree;
    tree.root = gtTree.root;
    tree.phaseDir = gtTree.phaseDir;
    tree.configPath = gtTree.configPath;
    tree.gtManifestPath = gtTree.phaseDir / "gt" / "gt_manifest.smoke.json";
    tree.sequence = gtTree.sequence;
    tree.frames = gtTree.frames;
    tree.keptPerFrame = gtTree.keptPerFrame;

    std::ostringstream config;
    config << "sequence: " << gtTree.sequence << "\n"
           << "dataset_root: " << gtTree.datasetRoot.string() << "\n"
           << "lidar_root: " << gtTree.lidarRoot.string() << "\n"
           << "cameras: [image_00, image_01]\n"
           << "artifacts:\n"
           << "  phase_dir: " << gtTree.phaseDir.string() << "\n"
           << "gt_cache:\n"
           << "  endianness: little\n"
           << "  record_size: 32\n"
           << "  min_points_per_frame: 1\n"
           << "  expected_points_range: [1, 100000]\n"
           << "  camera_unknown: 255\n"
           << "rays:\n"
           << "  t_min_m: 1.0e-4\n"
           << "  r_min_m: 1.5\n"
           << "  r_max_m: 60.0\n"
           << "  image_margin_px: 5\n"
           << "  camera_assignment: most_central_normalized_radius\n"
           << "  camera_tie_break: lower_camera_id\n"
           << "  buckets_primary: [[0.0, 20.0], [20.0, 40.0], [40.0, 50.0]]\n"
           << "  buckets_secondary: [[50.0, 60.0]]\n"
           << "  bucket_edge_rule: closed_right_for_40_50_and_50_60\n"
           << "  occlusion:\n"
           << "    neighbor_px: 3\n"
           << "    sigma_lidar_m: 0.02\n"
           << "    sigma_multiple: 3.0\n"
           << "    depth_threshold_m: 0.06\n"
           << "    variants: [occluded_excluded, full]\n"
           << "  record_size: 64\n";
    WriteGtTextFile(tree.configPath, config.str());
    return tree;
}

int RunSyntheticGtCache(const SyntheticRaysTree &tree)
{
    std::ostringstream frames;
    for (std::size_t index = 0; index < tree.frames.size(); ++index)
    {
        if (index != 0U)
        {
            frames << ',';
        }
        frames << tree.frames[index];
    }
    const std::vector<std::string> arguments = {"--config", tree.configPath.string(), "--frames",
                                                frames.str()};
    std::vector<std::string_view> views;
    views.reserve(arguments.size());
    for (const std::string &argument : arguments)
    {
        views.emplace_back(argument);
    }
    return ::RunGtCache(std::span<const std::string_view>(views.data(), views.size()));
}
} // namespace raylidar::test
