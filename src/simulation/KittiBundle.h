#pragma once

#include "LidarSensor.h"
#include "rt/TraceTypes.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace raylidar::simulation
{
using Matrix4d = std::array<std::array<double, 4>, 4>;

// Ground-truth rays from gt_rays.bin, shifted into the local frame.
struct GtRays
{
    std::vector<rt::Ray> Rays;
    std::vector<float> RangeGt;
    std::vector<std::uint32_t> Frames;
};

struct VelodynePose
{
    std::uint32_t Frame = 0;
    Matrix4d WorldFromVelodyne{};
};

// Reads the RLGR v1 format: 32-byte header, then 96-byte records
// (ray_id u64 | origin_W f64[3] | dir_W f64[3] | t_gt f64 | t_min f64 | t_max f64 | frame u32 | camera u16 | pad).
[[nodiscard]]
GtRays LoadGtRays(const std::filesystem::path &path, const std::array<double, 3> &origin);

// Reads frames[].T_W_V from camera_params.json.
[[nodiscard]]
std::vector<VelodynePose> LoadVelodynePoses(const std::filesystem::path &cameraParamsPath);

// Keeps the rotation and subtracts origin from the translation.
[[nodiscard]]
Transform ToLocalTransform(const Matrix4d &worldFromSensor, const std::array<double, 3> &origin);
} // namespace raylidar::simulation
