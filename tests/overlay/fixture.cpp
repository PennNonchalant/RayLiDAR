#include "fixture.h"

#include <stb_image.h>
#include <stb_image_write.h>

#include <algorithm>
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
#include <vector>

namespace raylidar::test
{
namespace
{
namespace fs = std::filesystem;

constexpr const char *kSequence = "2013_05_28_drive_0000_sync";
constexpr const char *kCameras[] = {"image_00", "image_01"};

std::atomic<unsigned long> gCounter{0};

std::string FrameName(int frame)
{
    std::ostringstream out;
    out << std::setw(10) << std::setfill('0') << frame;
    return out.str();
}

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

void WriteU32BigEndian(std::vector<unsigned char> &out, std::uint32_t value)
{
    out.push_back(static_cast<unsigned char>((value >> 24U) & 0xFFU));
    out.push_back(static_cast<unsigned char>((value >> 16U) & 0xFFU));
    out.push_back(static_cast<unsigned char>((value >> 8U) & 0xFFU));
    out.push_back(static_cast<unsigned char>(value & 0xFFU));
}

fs::path ImagePath(const fs::path &root, const std::string &camera, int frame, const std::string &subdir)
{
    return root / "data_2d_raw" / kSequence / camera / subdir / (FrameName(frame) + ".png");
}

void WriteCalibration(const fs::path &root, const OverlayTreeOptions &options)
{
    const fs::path dir = root / "calibration";
    {
        std::string text;
        text += "image_00: 1 0 0 0 0 1 0 0 0 0 1 0\n";
        text += "image_01: 1 0 0 0 0 1 0 0.06 0 0 1 0\n";
        WriteText(dir / "calib_cam_to_pose.txt", text);
    }
    WriteText(dir / "calib_cam_to_velo.txt", "1 0 0 0.10 0 1 0 0 0 0 1 -0.30\n");
    {
        const double cx = static_cast<double>(options.imageWidth) / 2.0;
        const double cy = static_cast<double>(options.imageHeight) / 2.0;
        std::ostringstream text;
        text.precision(12);
        text << "calib_time: fixture\n";
        text << "P_rect_00: 60 0 " << cx << " 0 0 60 " << cy << " 0 0 0 1 0\n";
        text << "R_rect_00: 1 0 0 0 1 0 0 0 1\n";
        text << "P_rect_01: 60 0 " << cx << " 0 0 60 " << cy << " 0 0 0 1 0\n";
        text << "R_rect_01: 1 0 0 0 1 0 0 0 1\n";
        WriteText(dir / "perspective.txt", text.str());
    }
}

void WritePoses(const fs::path &root, const OverlayTreeOptions &options)
{
    std::ostringstream text;
    text.precision(12);
    text << options.frame << " 1 0 0 " << options.frame << " 0 1 0 0 0 0 1 0\n";
    WriteText(root / "data_poses" / kSequence / "poses.txt", text.str());
}

void WriteCam0ToWorld(const fs::path &root, const OverlayTreeOptions &options)
{
    std::ostringstream text;
    text.precision(12);
    text << options.frame << " 1 0 0 " << options.frame << " 0 1 0 0 0 0 1 0 0 0 0 1\n";
    WriteText(root / "data_poses" / kSequence / "cam0_to_world.txt", text.str());
}

void WriteLidar(const fs::path &root, const OverlayTreeOptions &options)
{
    const fs::path path =
        root / "data_3d_raw" / kSequence / "velodyne_points" / "data" / (FrameName(options.frame) + ".bin");
    if (options.omitLidar)
    {
        return;
    }
    if (options.malformedLidar)
    {
        WriteBytes(path, std::vector<unsigned char>(10U, 0U));
        return;
    }
    const std::vector<float> values = OverlayScanPoints(options.pointCount);
    std::vector<unsigned char> bytes(values.size() * sizeof(float));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    WriteBytes(path, bytes);
}

void WriteImages(const fs::path &root, const OverlayTreeOptions &options)
{
    if (options.omitImage)
    {
        return;
    }
    const int x0 = options.imageWidth * 3 / 8;
    const int y0 = options.imageHeight * 3 / 8;
    const int x1 = options.imageWidth * 5 / 8;
    const int y1 = options.imageHeight * 5 / 8;
    const std::vector<std::uint8_t> rgb =
        RgbImageWithBlackRect(options.imageWidth, options.imageHeight, x0, y0, x1, y1);

    for (const char *camera : kCameras)
    {
        if (options.linkRectToRaw)
        {
            const fs::path rawPath = ImagePath(root, camera, options.frame, "data_rgb");
            WriteRgbPng(rawPath, options.imageWidth, options.imageHeight, rgb);
            const fs::path rectPath = ImagePath(root, camera, options.frame, "data_rect");
            EnsureParent(rectPath);
            std::error_code error;
            fs::remove(rectPath, error);
            fs::create_symlink(fs::path("..") / "data_rgb" / (FrameName(options.frame) + ".png"), rectPath, error);
            if (error)
            {
                throw std::runtime_error("cannot create fixture symlink: " + rectPath.string());
            }
        }
        else
        {
            WriteRgbPng(ImagePath(root, camera, options.frame, "data_rect"), options.imageWidth, options.imageHeight,
                        rgb);
        }
    }
}
} // namespace

Eigen::Matrix3d OverlayIntrinsics(double fx, double fy, double cx, double cy)
{
    Eigen::Matrix3d K = Eigen::Matrix3d::Zero();
    K(0, 0) = fx;
    K(1, 1) = fy;
    K(0, 2) = cx;
    K(1, 2) = cy;
    K(2, 2) = 1.0;
    return K;
}

Eigen::Matrix4d Translation4(double x, double y, double z)
{
    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
    transform(0, 3) = x;
    transform(1, 3) = y;
    transform(2, 3) = z;
    return transform;
}

std::vector<std::uint8_t> RgbImage(int width, int height, std::uint8_t value)
{
    return std::vector<std::uint8_t>(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3U, value);
}

std::vector<std::uint8_t> RgbImageWithBlackRect(int width, int height, int x0, int y0, int x1, int y1)
{
    std::vector<std::uint8_t> rgb = RgbImage(width, height, 255U);
    for (int y = std::max(y0, 0); y < std::min(y1, height); ++y)
    {
        for (int x = std::max(x0, 0); x < std::min(x1, width); ++x)
        {
            const std::size_t index =
                (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x)) * 3U;
            rgb[index + 0U] = 0U;
            rgb[index + 1U] = 0U;
            rgb[index + 2U] = 0U;
        }
    }
    return rgb;
}

std::vector<std::uint8_t> GrayImageWithBlackRect(int width, int height, int x0, int y0, int x1, int y1)
{
    std::vector<std::uint8_t> gray(static_cast<std::size_t>(width) * static_cast<std::size_t>(height), 255U);
    for (int y = std::max(y0, 0); y < std::min(y1, height); ++y)
    {
        for (int x = std::max(x0, 0); x < std::min(x1, width); ++x)
        {
            gray[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x)] = 0U;
        }
    }
    return gray;
}

void WriteRgbPng(const fs::path &path, int width, int height, const std::vector<std::uint8_t> &rgb)
{
    EnsureParent(path);
    if (stbi_write_png(path.string().c_str(), width, height, 3, rgb.data(), width * 3) == 0)
    {
        throw std::runtime_error("cannot write fixture PNG: " + path.string());
    }
}

void WritePngWithDimensions(const fs::path &path, int width, int height)
{
    EnsureParent(path);
    std::vector<unsigned char> bytes = {0x89U, 0x50U, 0x4EU, 0x47U, 0x0DU, 0x0AU, 0x1AU, 0x0AU};
    WriteU32BigEndian(bytes, 13U); // IHDR length
    bytes.push_back('I');
    bytes.push_back('H');
    bytes.push_back('D');
    bytes.push_back('R');
    WriteU32BigEndian(bytes, static_cast<std::uint32_t>(width));
    WriteU32BigEndian(bytes, static_cast<std::uint32_t>(height));
    bytes.push_back(8U);          // bit depth
    bytes.push_back(2U);          // colour type: RGB
    bytes.push_back(0U);          // compression
    bytes.push_back(0U);          // filter
    bytes.push_back(0U);          // interlace
    WriteU32BigEndian(bytes, 0U); // CRC is not validated by stbi_info
    // A header scan stops at IEND; adding it lets stbi_info report the size
    // without allocating or decoding any pixel data.
    WriteU32BigEndian(bytes, 0U); // IEND length
    bytes.push_back('I');
    bytes.push_back('E');
    bytes.push_back('N');
    bytes.push_back('D');
    WriteU32BigEndian(bytes, 0U); // IEND CRC

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output)
    {
        throw std::runtime_error("cannot write fixture PNG: " + path.string());
    }
    output.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

std::vector<std::uint8_t> ReadRgbPng(const fs::path &path, int &width, int &height)
{
    int channels = 0;
    unsigned char *data = stbi_load(path.string().c_str(), &width, &height, &channels, 3);
    if (data == nullptr)
    {
        throw std::runtime_error("cannot read fixture PNG: " + path.string());
    }
    std::vector<std::uint8_t> rgb(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3U);
    std::memcpy(rgb.data(), data, rgb.size());
    stbi_image_free(data);
    return rgb;
}

std::filesystem::path MakeOverlayTempRoot(std::string_view name)
{
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const unsigned long counter = gCounter.fetch_add(1U);
    const fs::path root = fs::temp_directory_path() / ("raylidar_overlay_" + std::string(name) + "_" +
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

std::vector<float> OverlayScanPoints(int count)
{
    std::vector<float> values;
    values.reserve(static_cast<std::size_t>(std::max(count, 0)) * 4U);
    for (int index = 0; index < count; ++index)
    {
        const double t = count > 1 ? static_cast<double>(index) / static_cast<double>(count - 1) : 0.5;
        const double x = -1.5 + 3.0 * t;
        const double y = -0.2 + 0.4 * std::sin(static_cast<double>(index) * 0.1);
        const double z = 11.0 + 18.5 * t;
        values.push_back(static_cast<float>(x));
        values.push_back(static_cast<float>(y));
        values.push_back(static_cast<float>(z));
        values.push_back(0.0F);
    }
    return values;
}

std::filesystem::path MakeOverlayTree(const std::filesystem::path &tmp, const OverlayTreeOptions &options)
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

std::filesystem::path WriteOverlayConfig(const std::filesystem::path &path, const std::filesystem::path &root,
                                         const std::string &sequence)
{
    std::ostringstream text;
    text << "dataset_root: " << root.string() << "\n";
    text << "sequence: " << sequence << "\n";
    text << "cameras: [image_00, image_01]\n";
    text << "pose_lookup:\n  mode: nearest\n  max_delta_frames: 5\n";
    text << "frame_sampling:\n  mode: contiguous_window\n  count: 3\n";
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
} // namespace raylidar::test
