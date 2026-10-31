#include "fixture.h"

#include <stb_image_write.h>

#include <atomic>
#include <cstddef>
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
} // namespace

fs::path MakeMaskingTempRoot(const std::string_view name)
{
    std::error_code error;
    const fs::path base = fs::temp_directory_path(error) / "raylidar_masking_tests";
    fs::create_directories(base, error);
    const unsigned long counter = gCounter.fetch_add(1U) + 1U;
    const fs::path root = base / (std::string(name) + "_" + std::to_string(counter));
    fs::remove_all(root, error);
    fs::create_directories(root, error);
    return root;
}

void WriteTextFile(const fs::path &path, const std::string_view text)
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
    WriteTextFile(path,
                  std::string_view(reinterpret_cast<const char *>(bytes.data()), bytes.size()));
}

void WriteGrayscalePng(const fs::path &path, const int width, const int height,
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

std::string MakeBBoxXml(const std::vector<BBoxSpec> &specs)
{
    std::ostringstream xml;
    xml << "<?xml version=\"1.0\"?>\n<opencv_storage>\n";
    for (std::size_t index = 0; index < specs.size(); ++index)
    {
        const BBoxSpec &spec = specs[index];
        const std::string tag = "object" + std::to_string(index + 1U);
        xml << '<' << tag << ">\n"
            << "  <index>" << index << "</index>\n"
            << "  <label>" << spec.label << "</label>\n"
            << "  <semanticId>" << spec.semanticId << "</semanticId>\n"
            << "  <timestamp>" << spec.timestamp << "</timestamp>\n"
            << "  <dynamic>" << (spec.dynamic ? 1 : 0) << "</dynamic>\n"
            << "  <transform type_id=\"opencv-matrix\">\n"
            << "    <rows>4</rows>\n"
            << "    <cols>4</cols>\n"
            << "    <dt>f</dt>\n"
            << "    <data>\n";
        for (int row = 0; row < 4; ++row)
        {
            for (int col = 0; col < 4; ++col)
            {
                xml << ' ' << std::setprecision(9) << spec.T_W_B(row, col);
            }
            xml << '\n';
        }
        xml << "    </data>\n"
            << "  </transform>\n"
            << "  <start_frame>" << spec.startFrame << "</start_frame>\n"
            << "  <end_frame>" << spec.endFrame << "</end_frame>\n"
            << "</" << tag << ">\n";
    }
    xml << "</opencv_storage>\n";
    return xml.str();
}

Eigen::Matrix4d TranslatedScaled(const double x, const double y, const double z,
                                const double scale)
{
    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
    transform(0, 0) = scale;
    transform(1, 1) = scale;
    transform(2, 2) = scale;
    transform(0, 3) = x;
    transform(1, 3) = y;
    transform(2, 3) = z;
    return transform;
}

SyntheticMaskTree MakeSyntheticMaskTree(const std::string_view name, const bool writeSemantics)
{
    SyntheticMaskTree tree;
    tree.root = MakeMaskingTempRoot(name);
    tree.datasetRoot = tree.root / "dataset";
    tree.lidarRoot = tree.root / "lidar";
    tree.bboxPath = tree.root / "bboxes.xml";
    tree.sequence = kSequence;
    tree.frames = {0, 1};
    tree.phaseDir = tree.root / "artifacts" / "02";

    const std::vector<std::array<float, 3>> points = {
        {0.0F, 0.0F, 10.0F}, {1.0F, 0.0F, 10.0F}, {0.0F, 1.0F, 10.0F}, {1.0F, 1.0F, 10.0F}};

    std::ostringstream timestamps;
    std::ostringstream poses;
    for (int frame = 0; frame < 3; ++frame)
    {
        timestamps << "2013-05-28 00:00:0" << frame << ".000000000\n";
        poses << frame << " 1 0 0 " << (0.1 * frame) << " 0 1 0 0 0 0 1 0\n";
    }
    WriteTextFile(tree.lidarRoot / "timestamps.txt", timestamps.str());
    WriteTextFile(tree.datasetRoot / "data_poses" / kSequence / "poses.txt", poses.str());

    WriteTextFile(tree.datasetRoot / "calibration" / "calib_cam_to_velo.txt",
                  "1 0 0 0 0 1 0 0 0 0 1 0\n");
    WriteTextFile(tree.datasetRoot / "calibration" / "calib_cam_to_pose.txt",
                  "image_00: 1 0 0 0 0 1 0 0 0 0 1 0\n"
                  "image_01: 1 0 0 0 0 1 0 0 0 0 1 0\n");
    WriteTextFile(tree.datasetRoot / "calibration" / "perspective.txt",
                  "P_rect_00: 10 0 4 0 0 10 4 0 0 0 1 0\n"
                  "P_rect_01: 10 0 4 0 0 10 4 0 0 0 1 0\n"
                  "R_rect_00: 1 0 0 0 1 0 0 0 1\n"
                  "R_rect_01: 1 0 0 0 1 0 0 0 1\n");

    for (const int frame : tree.frames)
    {
        std::vector<std::array<float, 4>> scan;
        scan.reserve(points.size());
        for (const auto &point : points)
        {
            scan.push_back({point[0], point[1], point[2], 1.0F});
        }
        const fs::path scanPath = tree.lidarRoot / "data" / (FrameName(frame) + ".bin");
        std::vector<std::uint8_t> bytes;
        bytes.reserve(scan.size() * 16U);
        for (const auto &point : scan)
        {
            const auto *raw = reinterpret_cast<const std::uint8_t *>(point.data());
            bytes.insert(bytes.end(), raw, raw + sizeof(float) * point.size());
        }
        WriteRawBytes(scanPath, bytes);
        tree.pointCount = scan.size();

        for (const char *camera : kCameras)
        {
            if (!writeSemantics && frame == tree.frames.back() && std::string(camera) == "image_01")
            {
                continue;
            }
            const std::vector<std::uint8_t> labels(
                static_cast<std::size_t>(kImageWidth) * static_cast<std::size_t>(kImageHeight), 7U);
            WriteGrayscalePng(tree.datasetRoot / "data_2d_semantics" / "train" / kSequence /
                                  camera / "semantic" / (FrameName(frame) + ".png"),
                              kImageWidth, kImageHeight, labels);
        }
    }

    WriteTextFile(tree.bboxPath, MakeBBoxXml({}));

    std::ostringstream config;
    config << "sequence: " << kSequence << "\n"
           << "dataset_root: " << tree.datasetRoot.string() << "\n"
           << "lidar_root: " << tree.lidarRoot.string() << "\n"
           << "bboxes_xml: " << tree.bboxPath.string() << "\n"
           << "cameras: [image_00, image_01]\n"
           << "artifacts:\n"
           << "  phase_dir: " << tree.phaseDir.string() << "\n"
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
    WriteTextFile(tree.configPath, config.str());
    return tree;
}
} // namespace raylidar::test
