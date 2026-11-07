#include "KittiBundle.h"

#include <nlohmann/json.hpp>

#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

namespace raylidar::simulation
{
namespace
{
constexpr std::size_t kGtHeaderSize = 32;
constexpr std::size_t kGtRecordSize = 96;

template <typename T> T Load(const char *data)
{
    T value;
    std::memcpy(&value, data, sizeof(T));
    return value;
}
} // namespace

GtRays LoadGtRays(const std::filesystem::path &path, const std::array<double, 3> &origin)
{
    std::ifstream file(path, std::ios::binary);

    if (!file)
    {
        throw std::runtime_error("Failed to open GT rays: " + path.string());
    }

    std::array<char, kGtHeaderSize> header{};
    file.read(header.data(), header.size());

    if (!file || std::memcmp(header.data(), "RLGR", 4) != 0 || Load<std::uint16_t>(header.data() + 4) != 1 ||
        Load<std::uint16_t>(header.data() + 6) != kGtRecordSize)
    {
        throw std::runtime_error("Unsupported GT rays header: " + path.string());
    }

    const auto count = static_cast<std::size_t>(Load<std::uint64_t>(header.data() + 8));

    std::vector<char> records(count * kGtRecordSize);
    file.read(records.data(), static_cast<std::streamsize>(records.size()));

    if (!file)
    {
        throw std::runtime_error("GT rays file is truncated: " + path.string());
    }

    GtRays result;
    result.Rays.resize(count);
    result.RangeGt.resize(count);
    result.Frames.resize(count);

    for (std::size_t index = 0; index < count; ++index)
    {
        const char *record = records.data() + index * kGtRecordSize;
        auto &ray = result.Rays[index];

        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            ray.Origin[axis] = static_cast<float>(Load<double>(record + 8 + axis * 8) - origin[axis]);
            ray.Direction[axis] = static_cast<float>(Load<double>(record + 32 + axis * 8));
        }

        result.RangeGt[index] = static_cast<float>(Load<double>(record + 56));
        ray.TMin = static_cast<float>(Load<double>(record + 64));
        ray.TMax = static_cast<float>(Load<double>(record + 72));
        result.Frames[index] = Load<std::uint32_t>(record + 80);
    }

    return result;
}

std::vector<VelodynePose> LoadVelodynePoses(const std::filesystem::path &cameraParamsPath)
{
    std::ifstream file(cameraParamsPath);

    if (!file)
    {
        throw std::runtime_error("Failed to open camera params: " + cameraParamsPath.string());
    }

    const auto json = nlohmann::json::parse(file);

    std::vector<VelodynePose> poses;

    for (const auto &frame : json.at("frames"))
    {
        VelodynePose pose;
        pose.Frame = frame.at("frame").get<std::uint32_t>();
        pose.WorldFromVelodyne = frame.at("T_W_V").get<Matrix4d>();
        poses.push_back(pose);
    }

    return poses;
}

Transform ToLocalTransform(const Matrix4d &worldFromSensor, const std::array<double, 3> &origin)
{
    Transform local{};

    for (std::size_t row = 0; row < 3; ++row)
    {
        for (std::size_t column = 0; column < 3; ++column)
        {
            local[row][column] = static_cast<float>(worldFromSensor[row][column]);
        }

        local[row][3] = static_cast<float>(worldFromSensor[row][3] - origin[row]);
    }

    return local;
}
} // namespace raylidar::simulation
