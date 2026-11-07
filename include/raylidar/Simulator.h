#pragma once

#include <raylidar/export.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace raylidar
{
// Defaults approximate the Velodyne HDL-64E used by KITTI-360. Sensor frame: +x forward, +y left, +z up.
struct SensorConfig
{
    std::uint32_t BeamCount = 64;
    std::uint32_t AzimuthCount = 2048;

    float ElevationMinDegrees = -24.8f;
    float ElevationMaxDegrees = 2.0f;

    float RangeMin = 0.0f;
    float RangeMax = 120.0f;
};

// Row-major 4x4 world-from-sensor transform, in the same world frame as the mesh (e.g. KITTI-360 T_W_V).
using Pose = std::array<std::array<double, 4>, 4>;

// Per-stage cost of one scan, in milliseconds.
struct ScanTimings
{
    double Generate = 0.0;   // CPU: build rays from the pose
    double Upload = 0.0;     // CPU: copy rays into the upload buffer
    double Trace = 0.0;      // GPU: ray query dispatch
    double CopyHits = 0.0;   // GPU: hit buffer -> readback buffer
    double SubmitWait = 0.0; // CPU: record, submit and wait for the GPU
    double Readback = 0.0;   // CPU: read hits and build points
    double Total = 0.0;      // CPU: the whole Scan() call
};

struct ScanOutput
{
    // World-space points of rays that hit the mesh.
    std::vector<std::array<double, 3>> Points;

    // One range per ray (index = beam * AzimuthCount + azimuth), in metres; negative when the ray missed.
    std::vector<float> Ranges;

    ScanTimings Timings;
};

// Mesh-based GPU LiDAR simulator. Loads the mesh once, then scans from any pose.
class Simulator
{
  public:
    // meshPath: binary little-endian PLY in world coordinates. traceShaderPath: compiled LidarTrace.spv.
    RAYLIDAR_EXPORT Simulator(const std::filesystem::path &meshPath, const std::filesystem::path &traceShaderPath,
                              const SensorConfig &sensor = {});
    RAYLIDAR_EXPORT ~Simulator();

    Simulator(const Simulator &) = delete;
    Simulator &operator=(const Simulator &) = delete;

    RAYLIDAR_EXPORT Simulator(Simulator &&) noexcept;
    RAYLIDAR_EXPORT Simulator &operator=(Simulator &&) noexcept;

    [[nodiscard]]
    RAYLIDAR_EXPORT ScanOutput Scan(const Pose &worldFromSensor);

    [[nodiscard]]
    RAYLIDAR_EXPORT std::uint32_t RayCount() const noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};
} // namespace raylidar
