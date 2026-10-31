#include "fixture.h"

#include <stb_image_write.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
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
    output.write(reinterpret_cast<const char *>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
}

void WritePng(const fs::path &path, int width, int height)
{
    EnsureParent(path);
    const std::size_t pixelCount =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    std::vector<unsigned char> pixels(pixelCount, 128U);
    if (stbi_write_png(path.string().c_str(), width, height, 1, pixels.data(), width) == 0)
    {
        throw std::runtime_error("cannot write fixture PNG: " + path.string());
    }
}

bool IsMissing(const DatasetTreeOptions &options, std::string_view token)
{
    return options.missing.find(std::string(token)) != options.missing.end();
}

void WriteCalibration(const fs::path &root, const DatasetTreeOptions &options)
{
    if (IsMissing(options, "calibration"))
    {
        return;
    }
    const fs::path directory = root / "calibration";
    if (!IsMissing(options, "calibration/calib_cam_to_pose.txt"))
    {
        WriteText(directory / "calib_cam_to_pose.txt",
                  "image_00: 1 0 0 0 0 1 0 0 0 0 1 0\n"
                  "image_01: 1 0 0 0 0 1 0 0 0 0 1 0\n");
    }
    if (!IsMissing(options, "calibration/calib_cam_to_velo.txt"))
    {
        WriteText(directory / "calib_cam_to_velo.txt", "1 0 0 0 0 1 0 0 0 0 1 0\n");
    }
    if (!IsMissing(options, "calibration/perspective.txt"))
    {
        WriteText(directory / "perspective.txt",
                  "P_rect_00: 552.55 0 682.05 0 0 552.55 188.0 0 0 0 1 0\n"
                  "R_rect_00: 1 0 0 0 1 0 0 0 1\n"
                  "P_rect_01: 552.55 0 682.05 0 0 552.55 188.0 0 0 0 1 0\n"
                  "R_rect_01: 1 0 0 0 1 0 0 0 1\n");
    }
}

void WritePoses(const fs::path &root, const DatasetTreeOptions &options)
{
    if (IsMissing(options, "poses"))
    {
        return;
    }
    const fs::path directory = root / "data_poses" / kSequence;
    if (!IsMissing(options, "poses/poses.txt"))
    {
        WriteText(directory / "poses.txt",
                  "0 1 0 0 0 0 1 0 0 0 0 1 0\n"
                  "1 1 0 0 0 0 1 0 0 0 0 1 0\n"
                  "2 1 0 0 0 0 1 0 0 0 0 1 0\n");
    }
    if (!IsMissing(options, "poses/cam0_to_world.txt"))
    {
        WriteText(directory / "cam0_to_world.txt",
                  "0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1\n"
                  "1 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1\n"
                  "2 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1\n");
    }
}

void WriteOfficialImages(const fs::path &root, const DatasetTreeOptions &options)
{
    if (IsMissing(options, "images"))
    {
        return;
    }
    const int counts[2] = {options.imageCounts.first, options.imageCounts.second};
    for (int index = 0; index < 2; ++index)
    {
        const fs::path directory =
            root / "data_2d_raw" / kSequence / kCameras[index] / "data_rect";
        for (int frame = 0; frame < counts[index]; ++frame)
        {
            char name[16];
            std::snprintf(name, sizeof(name), "%010d.png", frame);
            WritePng(directory / name, options.imageSize.first, options.imageSize.second);
        }
    }
}

void WriteSemantics(const fs::path &root, const DatasetTreeOptions &options)
{
    if (IsMissing(options, "semantics"))
    {
        return;
    }
    const int counts[2] = {options.imageCounts.first, options.imageCounts.second};
    for (int index = 0; index < 2; ++index)
    {
        const fs::path directory =
            root / "data_2d_semantics" / "train" / kSequence / kCameras[index] / "semantic";
        for (int frame = 0; frame < counts[index]; ++frame)
        {
            char name[16];
            std::snprintf(name, sizeof(name), "%010d.png", frame);
            WritePng(directory / name, options.imageSize.first, options.imageSize.second);
        }
    }
}

void WriteLidar(const fs::path &root, const DatasetTreeOptions &options)
{
    if (IsMissing(options, "lidar"))
    {
        return;
    }
    const fs::path directory =
        root / "data_3d_raw" / kSequence / "velodyne_points" / "data";
    WriteBytes(directory / "0000000000.bin", std::vector<unsigned char>(16U * 4U, 0U));
}

void WriteRepackagedImages(const fs::path &root, const DatasetTreeOptions &options)
{
    if (IsMissing(options, "images"))
    {
        return;
    }
    const int counts[2] = {options.imageCounts.first, options.imageCounts.second};
    for (int index = 0; index < 2; ++index)
    {
        const fs::path directory =
            root / "raw" / "2013_05_28" / kSequence / kCameras[index] / "data";
        for (int frame = 0; frame < counts[index]; ++frame)
        {
            char name[16];
            std::snprintf(name, sizeof(name), "%010d.png", frame);
            WritePng(directory / name, options.imageSize.first, options.imageSize.second);
        }
    }
}

void WriteKittiRawCalibration(const fs::path &root)
{
    const fs::path directory = root / "raw" / "2013_05_28";
    WriteText(directory / "calib_cam_to_cam.txt", "P_rect_00: 0 0 0 0\n");
    WriteText(directory / "calib_velo_to_cam.txt", "R: 1 0 0 0 1 0 0 0 1\n");
    WriteText(directory / "calib_imu_to_velo.txt", "R: 1 0 0 0 1 0 0 0 1\n");
}

} // namespace

std::filesystem::path MakeTempRoot(std::string_view name)
{
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const unsigned long counter = gCounter.fetch_add(1U);
    const fs::path root = fs::temp_directory_path() /
                          ("raylidar_" + std::string(name) + "_" + std::to_string(stamp) + "_" +
                           std::to_string(counter));
    std::error_code error;
    fs::remove_all(root, error);
    fs::create_directories(root, error);
    if (error)
    {
        throw std::runtime_error("cannot create fixture root: " + root.string());
    }
    return root;
}

std::filesystem::path MakeDatasetTree(const std::filesystem::path &tmp,
                                      const DatasetTreeOptions &options)
{
    std::error_code error;
    fs::remove_all(tmp, error);
    fs::create_directories(tmp, error);
    if (error)
    {
        throw std::runtime_error("cannot reset fixture root: " + tmp.string());
    }

    if (options.official)
    {
        WriteCalibration(tmp, options);
        WritePoses(tmp, options);
        WriteOfficialImages(tmp, options);
        WriteSemantics(tmp, options);
        WriteLidar(tmp, options);
    }

    if (options.repackaged)
    {
        WriteRepackagedImages(tmp, options);
    }

    if (options.kittiRawCalib)
    {
        WriteKittiRawCalibration(tmp);
    }

    return tmp;
}

std::string ReadFixtureFile(const std::filesystem::path &path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input)
    {
        throw std::runtime_error("cannot read fixture file: " + path.string());
    }
    return std::string(std::istreambuf_iterator<char>(input),
                       std::istreambuf_iterator<char>());
}
} // namespace raylidar::test
