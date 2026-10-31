#include "fixture.h"

#include <raylidar/common/mask_bits.h>
#include <raylidar/lidar/uncurl.h>

#include <Eigen/Core>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

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

namespace
{
constexpr int kAzimuthBins = 72;
constexpr int kRadialReturns = 120;
constexpr double kPi = 3.141592653589793238462643383279;
constexpr double kMotionM = 1.0; // metres per 1 s sweep

std::string FrameName(const int frame)
{
    char buffer[16] = {};
    std::snprintf(buffer, sizeof(buffer), "%010d", frame);
    return std::string(buffer);
}

Eigen::Vector3d StaticWorldPoint(const int azimuthBin, const int radial)
{
    const double thetaDeg = -180.0 + (static_cast<double>(azimuthBin) + 0.5) * 5.0;
    const double theta = thetaDeg * kPi / 180.0;
    const double radius = 20.0 + 0.02 * static_cast<double>(radial);
    return {radius * std::cos(theta), radius * std::sin(theta), 0.5};
}

std::array<float, 4> RawPointForWorld(const Eigen::Vector3d &world, const int frame)
{
    // Solve u = AzimuthFraction(P - (frame + u) * motion) by fixed-point
    // iteration; the contraction factor is motion / radius < 0.1 here.
    double u = 0.5;
    Eigen::Vector3d raw = world;
    for (int iteration = 0; iteration < 50; ++iteration)
    {
        raw = world;
        raw.x() -= (static_cast<double>(frame) + u) * kMotionM;
        u = raylidar::lidar::AzimuthFraction(raw);
    }
    return {static_cast<float>(raw.x()), static_cast<float>(raw.y()),
            static_cast<float>(raw.z()), 1.0F};
}

void WriteCalibration(const fs::path &datasetRoot)
{
    const fs::path calib = datasetRoot / "calibration";
    WriteTextFile(calib / "calib_cam_to_velo.txt", "1 0 0 0 0 1 0 0 0 0 1 0\n");
    WriteTextFile(calib / "calib_cam_to_pose.txt",
                  "image_00: 1 0 0 0 0 1 0 0 0 0 1 0\n");
    WriteTextFile(calib / "perspective.txt",
                  "P_rect_00: 1000 0 704 0 0 1000 188 0 0 0 1 0\n"
                  "R_rect_00: 1 0 0 0 1 0 0 0 1\n");
}

void WritePoses(const fs::path &datasetRoot, const std::string &sequence)
{
    std::ostringstream poses;
    for (int frame = 0; frame < 3; ++frame)
    {
        poses << frame << " 1 0 0 " << frame << " 0 1 0 0 0 0 1 0\n";
    }
    WriteTextFile(datasetRoot / "data_poses" / sequence / "poses.txt", poses.str());
}

std::string MakeConfig(const SyntheticUncurlTree &tree)
{
    std::ostringstream config;
    config << "sequence: 2013_05_28_drive_0000_sync\n"
           << "dataset_root: " << tree.datasetRoot.string() << "\n"
           << "lidar_root: " << tree.lidarRoot.string() << "\n"
           << "artifacts:\n"
           << "  phase_dir: " << (tree.root / "artifacts" / "02").string() << "\n"
           << "uncurl:\n"
           << "  azimuth_convention: 0.5_plus_atan2_over_2pi\n"
           << "  interval: current_to_next_timestamp\n"
           << "  max_next_pose_delta_frames: 0\n"
           << "  azimuth_bins_deg: 5\n"
           << "  match_voxel_m: 0.2\n"
           << "  max_match_distance_m: 0.3\n"
           << "  max_abs_bin_mean_m: 0.05\n"
           << "  max_sinusoid_amplitude_m: 0.03\n"
           << "  require_ablation_reported: true\n";
    return config.str();
}
} // namespace

void WriteSyntheticMask(const fs::path &maskDir, const int frame, const std::size_t pointCount,
                        const int dynamicIndex)
{
    std::vector<std::uint8_t> bits(
        pointCount, static_cast<std::uint8_t>(raylidar::masking::kKeptStatic));
    if (dynamicIndex >= 0 && static_cast<std::size_t>(dynamicIndex) < pointCount)
    {
        bits[static_cast<std::size_t>(dynamicIndex)] =
            static_cast<std::uint8_t>(raylidar::masking::kDynamicUnion);
    }
    WriteRawBytes(maskDir / ("frame_" + FrameName(frame)) / "points.bin", bits);
}

SyntheticUncurlTree MakeSyntheticUncurlTree(const std::string_view name, const bool writeMasks)
{
    SyntheticUncurlTree tree;
    tree.root = MakeLidarTempRoot(name);
    tree.datasetRoot = tree.root;
    tree.lidarRoot = tree.root / "lidar";
    tree.maskDir = tree.root / "masks";
    tree.outDir = tree.root / "out";
    tree.frames = {0, 1};
    tree.staticPointCount = static_cast<std::size_t>(kAzimuthBins * kRadialReturns);
    tree.totalPointCount = tree.staticPointCount + 1U;
    constexpr const char *kSequence = "2013_05_28_drive_0000_sync";

    for (const int frame : tree.frames)
    {
        std::vector<std::array<float, 4>> points;
        points.reserve(tree.totalPointCount);
        for (int bin = 0; bin < kAzimuthBins; ++bin)
        {
            for (int radial = 0; radial < kRadialReturns; ++radial)
            {
                points.push_back(RawPointForWorld(StaticWorldPoint(bin, radial), frame));
            }
        }
        points.push_back(RawPointForWorld(Eigen::Vector3d(25.0, 2.0, 0.5), frame));
        WriteVelodyneBin(tree.lidarRoot / "data" / (FrameName(frame) + ".bin"), points);
        if (writeMasks)
        {
            WriteSyntheticMask(tree.maskDir, frame, points.size(),
                               static_cast<int>(points.size() - 1U));
        }
    }

    WriteTextFile(tree.lidarRoot / "timestamps.txt",
                  "2013-05-28 00:00:00.000000000\n"
                  "2013-05-28 00:00:01.000000000\n"
                  "2013-05-28 00:00:02.000000000\n");
    WriteCalibration(tree.datasetRoot);
    WritePoses(tree.datasetRoot, kSequence);

    tree.configPath = tree.root / "phase2.yaml";
    WriteTextFile(tree.configPath, MakeConfig(tree));
    return tree;
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
