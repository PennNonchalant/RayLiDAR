#include "fixture.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace raylidar::test
{
namespace
{
namespace fs = std::filesystem;

constexpr const char *kSequence = "2013_05_28_drive_0000_sync";
constexpr const char *kCameras[] = {"image_00", "image_01"};
constexpr double kPi = 3.14159265358979323846;

std::atomic<unsigned long> gCounter{0};

void EnsureParent(const fs::path &path)
{
    std::error_code error;
    fs::create_directories(path.parent_path(), error);
}

void WriteText(const fs::path &path, std::string_view text)
{
    EnsureParent(path);
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output)
    {
        throw std::runtime_error("cannot write fixture file: " + path.string());
    }
    output.write(text.data(), static_cast<std::streamsize>(text.size()));
}

void WriteBytes(const fs::path &path, const std::vector<unsigned char> &bytes)
{
    EnsureParent(path);
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output)
    {
        throw std::runtime_error("cannot write fixture file: " + path.string());
    }
    output.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

std::string FrameName(int frame)
{
    std::ostringstream out;
    out << std::setw(10) << std::setfill('0') << frame;
    return out.str();
}

std::string Format(double value)
{
    std::ostringstream out;
    out.precision(12);
    out << value;
    return out.str();
}

std::string Matrix12(const double *values)
{
    std::string row;
    for (int index = 0; index < 12; ++index)
    {
        if (index != 0)
        {
            row += ' ';
        }
        row += Format(values[index]);
    }
    return row;
}

void WriteCalibration(const fs::path &root, const CalibTreeOptions &options)
{
    const fs::path dir = root / "calibration";

    {
        std::string text;
        text += "image_00: 1 0 0 0 0 1 0 0 0 0 1 0\n";
        text += "image_01: 1 0 0 " + Format(options.cam1Offset.x()) + " 0 1 0 " + Format(options.cam1Offset.y()) +
                " 0 0 1 " + Format(options.cam1Offset.z()) + "\n";
        WriteText(dir / "calib_cam_to_pose.txt", text);
    }
    {
        const double values[12] = {1.0, 0.0, 0.0, 0.10, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, -0.30};
        WriteText(dir / "calib_cam_to_velo.txt", Matrix12(values) + "\n");
    }
    {
        // First-order perturbation keeps det/orthonormality errors at ~eps.
        const double eps = options.rectPerturbation;
        const double rectification[9] = {1.0, -eps, 0.0, eps, 1.0, 0.0, 0.0, 0.0, 1.0};
        std::ostringstream text;
        text.precision(12);
        text << "calib_time: 17-Feb-2016 14:23:48\n";
        text << "P_rect_00: 700 0 704 0 0 700 188 0 0 0 1 0\n";
        text << "R_rect_00:";
        for (const double value : rectification)
        {
            text << ' ' << value;
        }
        text << "\n";
        text << "P_rect_01: 700 0 704 -328.318735 0 700 188 0 0 0 1 0\n";
        text << "R_rect_01: 1 0 0 0 1 0 0 0 1\n";
        WriteText(dir / "perspective.txt", text.str());
    }
}

void WritePoses(const fs::path &root, const CalibTreeOptions &options)
{
    std::string text;
    for (const int frame : options.frames)
    {
        if (options.omitPoses.find(frame) != options.omitPoses.end())
        {
            continue;
        }
        const double tx = static_cast<double>(frame);
        const double values[12] = {1.0, 0.0, 0.0, tx, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0};
        text += std::to_string(frame) + " " + Matrix12(values) + "\n";
    }
    WriteText(root / "data_poses" / kSequence / "poses.txt", text);
}

void WriteCam0ToWorld(const fs::path &root, const CalibTreeOptions &options)
{
    std::string text;
    for (const int frame : options.frames)
    {
        if (options.omitCam0.find(frame) != options.omitCam0.end())
        {
            continue;
        }
        const double tx = static_cast<double>(frame);
        const double values[16] = {1.0, 0.0, 0.0, tx, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0};
        text += std::to_string(frame);
        for (const double value : values)
        {
            text += ' ';
            text += Format(value);
        }
        text += "\n";
    }
    WriteText(root / "data_poses" / kSequence / "cam0_to_world.txt", text);
}

std::vector<float> GroundPoints(const CalibTreeOptions &options)
{
    std::vector<float> points;
    const auto push = [&points](double x, double y, double z) {
        points.push_back(static_cast<float>(x));
        points.push_back(static_cast<float>(y));
        points.push_back(static_cast<float>(z));
        points.push_back(0.0F);
    };

    for (int index = 0; index < 500; ++index)
    {
        const double angle = 2.0 * kPi * static_cast<double>(index) / 500.0;
        const double radius = 5.0 + 2.5 * static_cast<double>(index % 10);
        const double x = radius * std::cos(angle);
        const double y = radius * std::sin(angle);
        const double z = options.slopedGround ? 0.02 * x - 1.80 : -1.80;
        push(x, y, z);
    }
    if (options.slopedGround)
    {
        // Wall/car outliers that the per-azimuth lowest-return filter must ignore.
        for (int index = 0; index < 200; ++index)
        {
            const double angle = 2.0 * kPi * static_cast<double>(index) / 200.0;
            const double radius = 6.0 + 2.0 * static_cast<double>(index % 12);
            const double x = radius * std::cos(angle);
            const double y = radius * std::sin(angle);
            const double z = 0.5 + 1.5 * static_cast<double>(index % 7) / 7.0;
            push(x, y, z);
        }
    }
    return points;
}

void WriteLidar(const fs::path &root, const CalibTreeOptions &options)
{
    const std::vector<float> values = GroundPoints(options);
    std::vector<unsigned char> bytes(values.size() * sizeof(float));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    for (const int frame : options.frames)
    {
        if (options.omitLidar.find(frame) != options.omitLidar.end())
        {
            continue;
        }
        WriteBytes(root / "data_3d_raw" / kSequence / "velodyne_points" / "data" / (FrameName(frame) + ".bin"), bytes);
    }
}

void WriteImages(const fs::path &root, const CalibTreeOptions &options)
{
    for (const int frame : options.frames)
    {
        if (options.omitImages.find(frame) != options.omitImages.end())
        {
            continue;
        }
        for (const char *camera : kCameras)
        {
            // FindCommonFrames only checks existence; the overlay plan owns image
            // decoding and will build real fixtures.
            WriteBytes(root / "data_2d_raw" / kSequence / camera / "data_rect" / (FrameName(frame) + ".png"), {});
        }
    }
}
} // namespace

std::filesystem::path MakeTempRoot(std::string_view name)
{
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const unsigned long counter = gCounter.fetch_add(1U);
    const fs::path root = fs::temp_directory_path() / ("raylidar_calib_" + std::string(name) + "_" +
                                                       std::to_string(stamp) + "_" + std::to_string(counter));
    std::error_code error;
    fs::remove_all(root, error);
    fs::create_directories(root, error);
    if (error)
    {
        throw std::runtime_error("cannot create fixture root: " + root.string());
    }
    return root;
}

std::filesystem::path MakeCalibTree(const std::filesystem::path &tmp, const CalibTreeOptions &options)
{
    std::error_code error;
    fs::create_directories(tmp, error);
    if (error)
    {
        throw std::runtime_error("cannot create fixture root: " + tmp.string());
    }
    WriteCalibration(tmp, options);
    WritePoses(tmp, options);
    WriteCam0ToWorld(tmp, options);
    WriteLidar(tmp, options);
    WriteImages(tmp, options);
    return tmp;
}

std::filesystem::path WriteFramesConfig(const std::filesystem::path &path, const std::filesystem::path &root,
                                        const std::string &sequence, const std::vector<std::string> &cameras)
{
    std::ostringstream text;
    text << "dataset_root: " << root.string() << "\n";
    text << "sequence: " << sequence << "\n";
    text << "cameras: [";
    for (std::size_t index = 0; index < cameras.size(); ++index)
    {
        if (index != 0)
        {
            text << ", ";
        }
        text << cameras[index];
    }
    text << "]\n";
    text << "pose_lookup:\n  mode: nearest\n  max_delta_frames: 5\n";
    text << "seg_frame:\n  origin: median_camera_center\n";
    text << "frame_sampling:\n  mode: contiguous_window\n  count: 3\n";
    text << "assertions:\n";
    text << "  roundtrip_atol: 1.0e-9\n";
    text << "  rotation_atol: 2.0e-6\n";
    text << "  rotation_det_atol: 2.0e-6\n";
    text << "  cam0_world_atol: 1.0e-3\n";
    text << "  camera_height_m: [1.4, 1.9]\n";
    text << "  camera_height_stability_m: 0.1\n";
    text << "ground_estimate:\n";
    text << "  radius_m: [5.0, 30.0]\n";
    text << "  z_range_m: [-3.0, 1.0]\n";
    text << "  azimuth_bins: 240\n";
    text << "  trim_fraction: 0.7\n";
    text << "  min_inliers: 100\n";
    text << "overlay:\n";
    text << "  min_range_m: 10.0\n";
    text << "  max_range_m: 30.0\n";
    text << "  min_frames: 3\n";
    text << "  min_features: 5\n";
    text << "  max_offset_px: 2.0\n";
    text << "  image_margin_px: 5\n";
    WriteText(path, text.str());
    return path;
}

std::string ReadFixtureFile(const std::filesystem::path &path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input)
    {
        throw std::runtime_error("cannot read fixture file: " + path.string());
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}
} // namespace raylidar::test
