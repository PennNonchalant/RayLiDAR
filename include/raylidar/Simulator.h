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
// Beams are spread evenly from ElevationMinDegrees (beam 0) to ElevationMaxDegrees; azimuth steps cover 360 degrees.
struct SensorConfig
{
    std::uint32_t BeamCount = 64;      // >= 1
    std::uint32_t AzimuthCount = 2048; // >= 1; BeamCount * AzimuthCount is capped at 16M rays

    float ElevationMinDegrees = -24.8f; // in [-90, 90] and <= ElevationMaxDegrees
    float ElevationMaxDegrees = 2.0f;   // in [-90, 90]

    float RangeMin = 0.0f;   // metres, >= 0
    float RangeMax = 120.0f; // metres, > RangeMin
};

// Which scene representation the simulator traces.
enum class SceneMode : std::uint8_t
{
    Mesh,     // triangle mesh only: fastest
    Gaussian, // 2D Gaussian splats only
    Hybrid    // mesh first; rays that miss the mesh are traced again against the Gaussians
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

// How rays are traced against 2D Gaussians (Gaussian and Hybrid modes).
struct GaussianTraceConfig
{
    float SigmaCutoff = 3.0f;    // splat extent in standard deviations, > 0
    float AlphaThreshold = 0.5f; // accumulated opacity that counts as a hit, in (0, 1)
};

struct SimulatorConfig
{
    SceneMode Mode = SceneMode::Mesh;

    // Binary little-endian PLYs in world coordinates.
    // MeshPath: required in Mesh and Hybrid modes, ignored in Gaussian mode.
    // GaussianPath: a 2DGS point_cloud.ply, required in Gaussian and Hybrid modes. Optional in Mesh mode, where it
    // only adds trust labels (the Gaussians are compared with the mesh once, at load time, and never traced).
    std::filesystem::path MeshPath;
    std::filesystem::path GaussianPath;

    // Folder with the compiled shaders: LidarTrace.spv (Mesh, Hybrid), GaussianTraceK16.spv (Gaussian, Hybrid) and,
    // optionally, LidarRayGen.spv. Empty means the current directory.
    std::filesystem::path ShaderDirectory;

    SensorConfig Sensor;
    GaussianTraceConfig Gaussian;

    // Mesh mode only: generate rays on the GPU (about 2x faster per scan) when LidarRayGen.spv is in
    // ShaderDirectory. Falls back to CPU ray generation otherwise; check Simulator::UsesGpuRayGeneration().
    bool PreferGpuRayGeneration = true;
};

// Row-major 4x4 world-from-sensor transform, in the same world frame as the scene (e.g. KITTI-360 T_W_V).
// Must be rigid: an orthonormal rotation, a translation, and a last row of 0 0 0 1.
using Pose = std::array<std::array<double, 4>, 4>;

// Per-stage cost of one scan, in milliseconds. A stage that did not run is 0.
// The cost of the simulation itself is RayGen + Trace. The other stages move data between the CPU and the GPU and
// depend on what the caller does with the result.
struct ScanTimings
{
    double Generate = 0.0;   // CPU: build rays from the pose
    double Upload = 0.0;     // CPU: copy rays into the upload buffer
    double RayGen = 0.0;     // GPU: ray generation dispatch
    double Trace = 0.0;      // GPU: ray query dispatches, summed over passes
    double CopyHits = 0.0;   // GPU: hit buffer -> readback buffer
    double SubmitWait = 0.0; // CPU: record, submit and wait for the GPU
    double Readback = 0.0;   // CPU: read hits back and build points
    double Total = 0.0;      // CPU: the whole Simulator::Scan() call, wall time
};

// One sweep. Per-ray arrays have RayCount() = BeamCount * AzimuthCount entries, index = beam * AzimuthCount +
// azimuth, so they reshape directly to a range image.
struct ScanOutput
{
    // World-space points of the rays that hit, in ray order (not per ray).
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

// GPU LiDAR simulator. Loads the scene once, then scans it from any pose.
//
// Errors are reported with exceptions: std::invalid_argument for invalid input (config, pose, missing files),
// std::runtime_error for unreadable files and GPU failures.
//
// Not thread-safe: run one Scan() at a time per Simulator. Separate Simulators are independent. A moved-from
// Simulator may only be destroyed or assigned to.
class Simulator
{
  public:
    // Validates the whole config before touching the GPU and reports every problem in one exception.
    RAYLIDAR_EXPORT explicit Simulator(const SimulatorConfig &config);

    // Mesh mode, kept for existing callers. traceShaderPath: compiled LidarTrace.spv; the other shaders are looked up
    // in the same folder.
    RAYLIDAR_EXPORT Simulator(const std::filesystem::path &meshPath, const std::filesystem::path &traceShaderPath,
                              const SensorConfig &sensor = {});

    RAYLIDAR_EXPORT ~Simulator();

    Simulator(const Simulator &) = delete;
    Simulator &operator=(const Simulator &) = delete;

    RAYLIDAR_EXPORT Simulator(Simulator &&) noexcept;
    RAYLIDAR_EXPORT Simulator &operator=(Simulator &&) noexcept;

    // Throws std::invalid_argument when worldFromSensor is not a finite rigid transform.
    [[nodiscard]]
    RAYLIDAR_EXPORT ScanOutput Scan(const Pose &worldFromSensor);

    [[nodiscard]]
    RAYLIDAR_EXPORT std::uint32_t RayCount() const noexcept;

    [[nodiscard]]
    RAYLIDAR_EXPORT SceneMode Mode() const noexcept;

    [[nodiscard]]
    RAYLIDAR_EXPORT bool UsesGpuRayGeneration() const noexcept;

    // Writes one sweep as a .rlf file (read it in Python with python/raylidar_frame.py).
    // Throws std::invalid_argument when the per-ray arrays do not match BeamCount * AzimuthCount.
    RAYLIDAR_EXPORT static void WriteFrame(const std::filesystem::path &path, const ScanOutput &output,
                                           std::uint32_t frameIndex = 0);

  private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};
} // namespace raylidar
