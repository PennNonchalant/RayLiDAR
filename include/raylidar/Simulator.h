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

// Which scene representation the simulator traces.
enum class SceneMode : std::uint8_t
{
    Mesh,     // triangle mesh only: fastest
    Gaussian, // 2D Gaussian splats only
    Hybrid    // mesh first; rays that miss the mesh are re-traced against the Gaussians
};

// Which representation produced a ray's hit.
enum class HitSource : std::uint8_t
{
    None,
    Mesh,
    Gaussian
};

// Whether the mesh and the 2D Gaussians agree where a ray hit the mesh. Untrusted hits are much more likely to be
// wrong (e.g. hit too early). Needs both a mesh and Gaussians; otherwise every hit is Unknown.
enum class SurfaceTrust : std::uint8_t
{
    Unknown,
    Trusted,
    Untrusted
};

struct SimulatorConfig
{
    SceneMode Mode = SceneMode::Mesh;

    // Binary little-endian PLYs in world coordinates. Mesh and Hybrid need MeshPath; Gaussian and Hybrid need
    // GaussianPath (a 2DGS point_cloud.ply). Giving both in Mesh mode adds trust labels without tracing Gaussians.
    std::filesystem::path MeshPath;
    std::filesystem::path GaussianPath;

    // Folder with the compiled shaders: LidarTrace.spv, LidarRayGen.spv (optional, enables GPU ray generation in
    // Mesh mode) and GaussianTraceK16.spv.
    std::filesystem::path ShaderDirectory;

    SensorConfig Sensor;

    float SigmaCutoff = 3.0f;    // Gaussian extent, in standard deviations
    float AlphaThreshold = 0.5f; // accumulated opacity that counts as a Gaussian hit
};

// Row-major 4x4 world-from-sensor transform, in the same world frame as the scene (e.g. KITTI-360 T_W_V).
using Pose = std::array<std::array<double, 4>, 4>;

// Per-stage cost of one scan, in milliseconds.
struct ScanTimings
{
    double Generate = 0.0;   // CPU: build rays from the pose (0 when rays are generated on the GPU)
    double Upload = 0.0;     // CPU: copy rays into the upload buffer
    double RayGen = 0.0;     // GPU: ray generation dispatch (0 when rays are generated on the CPU)
    double Trace = 0.0;      // GPU: ray query dispatch
    double CopyHits = 0.0;   // GPU: hit buffer -> readback buffer
    double SubmitWait = 0.0; // CPU: record, submit and wait for the GPU
    double Readback = 0.0;   // CPU: read hits and build points
    double Total = 0.0;      // CPU: the whole Scan() call
};

// One sweep. Per-ray arrays have one entry per ray, index = beam * AzimuthCount + azimuth (a range image).
struct ScanOutput
{
    // World-space points of rays that hit, in ray order.
    std::vector<std::array<double, 3>> Points;

    // Per ray, in metres; negative when the ray missed.
    std::vector<float> Ranges;

    // Per ray: hit point in the sensor frame, 0 when the ray missed.
    std::vector<std::array<float, 3>> SensorPoints;

    // Per ray.
    std::vector<HitSource> Sources;
    std::vector<SurfaceTrust> Trust;

    Pose WorldFromSensor{};
    std::uint32_t BeamCount = 0;
    std::uint32_t AzimuthCount = 0;

    ScanTimings Timings;
};

// GPU LiDAR simulator. Loads the scene once, then scans from any pose.
class Simulator
{
  public:
    RAYLIDAR_EXPORT explicit Simulator(const SimulatorConfig &config);

    // Mesh only. meshPath: binary little-endian PLY in world coordinates. traceShaderPath: compiled LidarTrace.spv;
    // other shaders are looked up next to it.
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

    [[nodiscard]]
    RAYLIDAR_EXPORT SceneMode Mode() const noexcept;

    // Writes one sweep as a .rlf file (read it in Python with python/raylidar_frame.py).
    RAYLIDAR_EXPORT static void WriteFrame(const std::filesystem::path &path, const ScanOutput &output,
                                           std::uint32_t frameIndex = 0);

  private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};
} // namespace raylidar
