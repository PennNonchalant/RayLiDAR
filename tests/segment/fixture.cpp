#include "fixture.h"

#include <raylidar/common/io.h>

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
constexpr int kImageWidth = 8;
constexpr int kImageHeight = 8;

std::atomic<unsigned long> gCounter{0};

std::string FrameName(const int frame)
{
    std::ostringstream out;
    out << std::setw(10) << std::setfill('0') << frame;
    return out.str();
}

void WritePng(const fs::path &path, const std::vector<std::uint8_t> &labels)
{
    std::error_code error;
    fs::create_directories(path.parent_path(), error);
    if (stbi_write_png(path.string().c_str(), kImageWidth, kImageHeight, 1, labels.data(),
                       kImageWidth) == 0)
    {
        throw std::runtime_error("cannot write PNG " + path.string());
    }
}

std::string MakeBoxXml(const double x, const double y, const double z)
{
    std::ostringstream xml;
    xml << "<?xml version=\"1.0\"?>\n<opencv_storage>\n";
    for (int object = 0; object < 2; ++object)
    {
        const std::string tag = "object" + std::to_string(object + 1);
        xml << '<' << tag << ">\n"
            << "  <index>" << object << "</index>\n"
            << "  <label>car</label>\n"
            << "  <semanticId>26</semanticId>\n"
            << "  <timestamp>0</timestamp>\n"
            << "  <dynamic>1</dynamic>\n"
            << "  <transform type_id=\"opencv-matrix\">\n"
            << "    <rows>4</rows>\n"
            << "    <cols>4</cols>\n"
            << "    <dt>f</dt>\n"
            << "    <data>\n"
            << "      1 0 0 " << (object == 0 ? 0.0 : x) << '\n'
            << "      0 1 0 " << (object == 0 ? 0.0 : y) << '\n'
            << "      0 0 1 " << (object == 0 ? 0.0 : z) << '\n'
            << "      0 0 0 1\n"
            << "    </data>\n"
            << "  </transform>\n"
            << "  <start_frame>0</start_frame>\n"
            << "  <end_frame>100000</end_frame>\n"
            << "</" << tag << ">\n";
    }
    xml << "</opencv_storage>\n";
    return xml.str();
}
} // namespace

fs::path MakeSegmentTempRoot(const std::string_view name)
{
    std::error_code error;
    const fs::path base = fs::temp_directory_path(error) / "raylidar_segment_tests";
    fs::create_directories(base, error);
    const unsigned long counter = gCounter.fetch_add(1U) + 1U;
    const fs::path root = base / (std::string(name) + "_" + std::to_string(counter));
    fs::remove_all(root, error);
    fs::create_directories(root, error);
    return root;
}

void WriteSegmentBytes(const fs::path &path, const std::vector<std::uint8_t> &bytes)
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

void WriteSegmentText(const fs::path &path, const std::string_view text)
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

void WriteSyntheticFrameIndex(const fs::path &path, const std::vector<int> &usableFrames,
                              const std::string_view sequence)
{
    nlohmann::json document;
    document["schema_version"] = 1;
    document["sequence"] = std::string(sequence);
    document["usable_frames"] = usableFrames;
    document["counts"] = {{"usable", usableFrames.size()}, {"dropped", 0},
                          {"total", usableFrames.size()}};
    raylidar::WriteJson(path, document);
}

SyntheticSegmentTree MakeSyntheticSegmentTree(const std::string_view name)
{
    SyntheticSegmentTree tree;
    tree.root = MakeSegmentTempRoot(name);
    tree.datasetRoot = tree.root / "dataset";
    tree.lidarRoot = tree.root / "lidar";
    tree.bboxPath = tree.root / "bboxes.xml";
    tree.sequence = kSequence;
    tree.phaseDir = tree.root / "artifacts" / "02";
    tree.frameIndexPath = tree.phaseDir / "streams" / "frame_index.json";
    tree.frames = {0, 1, 2, 3, 4, 5};
    tree.points = {{0.1F, 0.1F, 0.0F, 1.0F},
                   {0.0F, 0.0F, 10.0F, 1.0F},
                   {5.0F, 0.0F, 10.0F, 1.0F},
                   {-5.0F, 0.0F, 10.0F, 1.0F}};

    std::ostringstream timestamps;
    std::ostringstream poses;
    for (int frame = 0; frame <= static_cast<int>(tree.frames.size()); ++frame)
    {
        timestamps << "2013-05-28 00:00:0" << frame << ".000000000\n";
        poses << frame << " 1 0 0 0 0 1 0 0 0 0 1 0\n";
    }
    WriteSegmentText(tree.lidarRoot / "timestamps.txt", timestamps.str());
    WriteSegmentText(tree.datasetRoot / "data_poses" / kSequence / "poses.txt", poses.str());

    WriteSegmentText(tree.datasetRoot / "calibration" / "calib_cam_to_velo.txt",
                     "1 0 0 0 0 1 0 0 0 0 1 0\n");
    WriteSegmentText(tree.datasetRoot / "calibration" / "calib_cam_to_pose.txt",
                     "image_00: 1 0 0 0 0 1 0 0 0 0 1 0\n"
                     "image_01: 1 0 0 0 0 1 0 0 0 0 1 0\n");
    WriteSegmentText(tree.datasetRoot / "calibration" / "perspective.txt",
                     "P_rect_00: 10 0 4 0 0 10 4 0 0 0 1 0\n"
                     "P_rect_01: 10 0 4 0 0 10 4 0 0 0 1 0\n"
                     "R_rect_00: 1 0 0 0 1 0 0 0 1\n"
                     "R_rect_01: 1 0 0 0 1 0 0 0 1\n");

    const std::vector<std::uint8_t> labels(
        static_cast<std::size_t>(kImageWidth) * static_cast<std::size_t>(kImageHeight), 7U);
    for (const int frame : tree.frames)
    {
        std::vector<std::uint8_t> scan;
        scan.reserve(tree.points.size() * 16U);
        for (const auto &point : tree.points)
        {
            const std::array<float, 4> values = {point.x, point.y, point.z, point.intensity};
            const auto *raw = reinterpret_cast<const std::uint8_t *>(values.data());
            scan.insert(scan.end(), raw, raw + sizeof(float) * values.size());
        }
        WriteSegmentBytes(tree.lidarRoot / "data" / (FrameName(frame) + ".bin"), scan);
        for (const char *camera : kCameras)
        {
            WritePng(tree.datasetRoot / "data_2d_semantics" / "train" / kSequence / camera /
                         "semantic" / (FrameName(frame) + ".png"),
                     labels);
        }
    }

    WriteSegmentText(tree.bboxPath, MakeBoxXml(0.0, 0.0, 10.0));

    std::ostringstream config;
    config << "sequence: " << kSequence << "\n"
           << "dataset_root: " << tree.datasetRoot.string() << "\n"
           << "lidar_root: " << tree.lidarRoot.string() << "\n"
           << "bboxes_xml: " << tree.bboxPath.string() << "\n"
           << "cameras: [image_00, image_01]\n"
           << "artifacts:\n"
           << "  phase_dir: " << tree.phaseDir.string() << "\n"
           << "selection:\n"
           << "  window_frames: 2\n"
           << "  step_frames: 1\n"
           << "  min_usable_frames: 2\n"
           << "  min_scene_points_per_frame: 0\n"
           << "  score: micro_dynamic_fraction_after_ego\n"
           << "  tie_break: earliest_window\n"
           << "mask:\n"
           << "  dynamic_classes: [car, truck, bus, motorcycle, bicycle, person, rider]\n"
           << "  dynamic_label_ids: [26, 27, 28, 32, 33, 24, 25]\n"
           << "  bbox_dynamic_label_map: {pedestrian: person}\n"
           << "  ego_roi_velodyne_m:\n"
           << "    x: [-2.5, 2.5]\n"
           << "    y: [-1.2, 1.2]\n"
           << "    z: [-2.0, 0.3]\n"
           << "  bbox_max_time_gap_frames: 2\n"
           << "  residual_dynamic_radius_m: 1.0\n";
    tree.configPath = tree.root / "phase2.yaml";
    WriteSegmentText(tree.configPath, config.str());

    WriteSyntheticFrameIndex(tree.frameIndexPath, tree.frames, tree.sequence);
    return tree;
}
} // namespace raylidar::test
