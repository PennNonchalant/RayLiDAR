#include "fixture.h"

#include <raylidar/common/mask_bits.h>

#include <stb_image_write.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace raylidar::test
{
namespace
{
namespace fs = std::filesystem;

constexpr const char *kSequence = "2013_05_28_drive_0000_sync";
constexpr const char *kCameras[] = {"image_00", "image_01"};
constexpr int kImageWidth = 64;
constexpr int kImageHeight = 48;

std::atomic<unsigned long> gCounter{0};

std::string FrameName(const int frame)
{
    std::ostringstream out;
    out << std::setw(10) << std::setfill('0') << frame;
    return out.str();
}

std::string MakePerspective()
{
    std::ostringstream text;
    text << "P_rect_00: 100 0 32 0 0 100 24 0 0 0 1 0\n"
         << "P_rect_01: 100 0 32 0 0 100 24 0 0 0 1 0\n"
         << "R_rect_00: 1 0 0 0 1 0 0 0 1\n"
         << "R_rect_01: 1 0 0 0 1 0 0 0 1\n";
    return text.str();
}

std::vector<std::uint8_t> EncodeScan(const std::vector<std::array<float, 4>> &points)
{
    std::vector<std::uint8_t> bytes;
    bytes.reserve(points.size() * 16U);
    for (const auto &point : points)
    {
        const auto *raw = reinterpret_cast<const std::uint8_t *>(point.data());
        bytes.insert(bytes.end(), raw, raw + sizeof(float) * point.size());
    }
    return bytes;
}
} // namespace

TwoCameraRig MakeTwoCameraRig()
{
    TwoCameraRig rig;
    rig.K << 100.0, 0.0, 200.0, 0.0, 100.0, 150.0, 0.0, 0.0, 1.0;
    for (const char *camera : kCameras)
    {
        rig.T_Ck_rect_V[camera] = Eigen::Matrix4d::Identity();
        rig.K_rect[camera] = rig.K;
        rig.imageSizes[camera] = {400, 300};
    }
    return rig;
}

Eigen::Matrix4d Translate(const double x, const double y, const double z)
{
    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
    transform(0, 3) = x;
    transform(1, 3) = y;
    transform(2, 3) = z;
    return transform;
}

fs::path MakeGtTempRoot(const std::string_view name)
{
    std::error_code error;
    const fs::path base = fs::temp_directory_path(error) / "raylidar_gt_tests";
    fs::create_directories(base, error);
    const unsigned long counter = gCounter.fetch_add(1U) + 1U;
    const fs::path root = base / (std::string(name) + "_" + std::to_string(counter));
    fs::remove_all(root, error);
    fs::create_directories(root, error);
    return root;
}

void WriteGtTextFile(const fs::path &path, const std::string_view text)
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

void WriteGtRawBytes(const fs::path &path, const std::vector<std::uint8_t> &bytes)
{
    WriteGtTextFile(path, std::string_view(reinterpret_cast<const char *>(bytes.data()), bytes.size()));
}

void WriteGtGrayscalePng(const fs::path &path, const int width, const int height,
                         const std::vector<std::uint8_t> &labels)
{
    std::error_code error;
    fs::create_directories(path.parent_path(), error);
    if (static_cast<std::size_t>(width) * static_cast<std::size_t>(height) != labels.size())
    {
        throw std::runtime_error("label buffer does not match image dimensions");
    }
    if (stbi_write_png(path.string().c_str(), width, height, 1, labels.data(), width) == 0)
    {
        throw std::runtime_error("cannot write PNG " + path.string());
    }
}

void WriteSyntheticSegmentManifest(const fs::path &path, const std::vector<int> &frames, const int frameCount,
                                   const std::string_view mode)
{
    std::ostringstream text;
    text << "{\n"
         << "  \"schema_version\": 1,\n"
         << "  \"sequence\": \"" << kSequence << "\",\n"
         << "  \"selection_mode\": \"" << mode << "\",\n"
         << "  \"selected\": {\n"
         << "    \"start_frame\": " << (frames.empty() ? 0 : frames.front()) << ",\n"
         << "    \"end_frame\": " << (frames.empty() ? 0 : frames.back()) << ",\n"
         << "    \"frame_count\": " << frameCount << ",\n"
         << "    \"frames\": [";
    for (std::size_t index = 0; index < frames.size(); ++index)
    {
        if (index != 0U)
        {
            text << ", ";
        }
        text << frames[index];
    }
    text << "]\n"
         << "  }\n"
         << "}\n";
    WriteGtTextFile(path, text.str());
}

SyntheticGtTree MakeSyntheticGtTree(const std::string_view name)
{
    SyntheticGtTree tree;
    tree.root = MakeGtTempRoot(name);
    tree.datasetRoot = tree.root / "dataset";
    tree.lidarRoot = tree.root / "lidar";
    tree.phaseDir = tree.root / "artifacts" / "02";
    tree.sequence = kSequence;
    tree.frames = {0, 1};

    const std::vector<std::array<float, 4>> points = {
        {0.0F, 0.0F, 10.0F, 1.0F}, {1.0F, 0.0F, 10.0F, 1.0F}, {0.0F, 1.0F, 10.0F, 1.0F}, {1.0F, 1.0F, 10.0F, 1.0F}};
    tree.pointCount = points.size();
    tree.keptPerFrame = 2;

    std::ostringstream timestamps;
    std::ostringstream poses;
    for (int frame = 0; frame < 3; ++frame)
    {
        timestamps << "2013-05-28 00:00:0" << frame << ".000000000\n";
        poses << frame << " 1 0 0 " << (0.1 * frame) << " 0 1 0 0 0 0 1 0\n";
    }
    tree.timestampsPath = tree.lidarRoot / "timestamps.txt";
    tree.posesPath = tree.datasetRoot / "data_poses" / kSequence / "poses.txt";
    WriteGtTextFile(tree.timestampsPath, timestamps.str());
    WriteGtTextFile(tree.posesPath, poses.str());

    WriteGtTextFile(tree.datasetRoot / "calibration" / "calib_cam_to_velo.txt", "1 0 0 0 0 1 0 0 0 0 1 0\n");
    WriteGtTextFile(tree.datasetRoot / "calibration" / "calib_cam_to_pose.txt", "image_00: 1 0 0 0 0 1 0 0 0 0 1 0\n"
                                                                                "image_01: 1 0 0 0 0 1 0 0 0 0 1 0\n");
    WriteGtTextFile(tree.datasetRoot / "calibration" / "perspective.txt", MakePerspective());

    const std::vector<std::uint8_t> labels(
        static_cast<std::size_t>(kImageWidth) * static_cast<std::size_t>(kImageHeight), 7U);

    const std::vector<std::uint8_t> maskBits = {static_cast<std::uint8_t>(raylidar::masking::kKeptStatic),
                                                static_cast<std::uint8_t>(raylidar::masking::kKeptStatic),
                                                static_cast<std::uint8_t>(raylidar::masking::kDynamicUnion),
                                                static_cast<std::uint8_t>(raylidar::masking::kEgoRoi)};

    for (const int frame : tree.frames)
    {
        WriteGtRawBytes(tree.lidarRoot / "data" / (FrameName(frame) + ".bin"), EncodeScan(points));
        for (const char *camera : kCameras)
        {
            WriteGtGrayscalePng(tree.datasetRoot / "data_2d_semantics" / "train" / kSequence / camera / "semantic" /
                                    (FrameName(frame) + ".png"),
                                kImageWidth, kImageHeight, labels);
        }
        WriteGtRawBytes(tree.phaseDir / "masks" / ("frame_" + FrameName(frame)) / "points.bin", maskBits);
    }

    tree.segmentManifestPath = tree.phaseDir / "segment" / "segment_manifest.json";
    WriteSyntheticSegmentManifest(tree.segmentManifestPath, tree.frames, static_cast<int>(tree.frames.size()));

    std::ostringstream config;
    config << "sequence: " << kSequence << "\n"
           << "dataset_root: " << tree.datasetRoot.string() << "\n"
           << "lidar_root: " << tree.lidarRoot.string() << "\n"
           << "bboxes_xml: " << (tree.root / "bboxes.xml").string() << "\n"
           << "cameras: [image_00, image_01]\n"
           << "artifacts:\n"
           << "  phase_dir: " << tree.phaseDir.string() << "\n"
           << "selection:\n"
           << "  window_frames: " << tree.frames.size() << "\n"
           << "  step_frames: 10\n"
           << "  min_usable_frames: " << tree.frames.size() << "\n"
           << "  min_scene_points_per_frame: 1\n"
           << "gt_cache:\n"
           << "  endianness: little\n"
           << "  record_size: 32\n"
           << "  min_points_per_frame: 20000\n"
           << "  expected_points_range: [50000, 150000]\n"
           << "  camera_unknown: 255\n"
           << "rays:\n"
           << "  t_min_m: 1.0e-4\n"
           << "  r_min_m: 1.5\n"
           << "  r_max_m: 60.0\n"
           << "  image_margin_px: 5\n";
    tree.configPath = tree.root / "phase2.yaml";
    WriteGtTextFile(tree.configPath, config.str());
    return tree;
}
} // namespace raylidar::test
