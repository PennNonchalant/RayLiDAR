#pragma once

#include "KittiBundle.h"
#include "LidarSensor.h"
#include "rt/TraceTypes.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace raylidar::simulation
{
// Which representation produced a ray's hit.
enum class HitSource : std::uint8_t
{
    None,
    Mesh,
    Gaussian
};

// Whether the mesh and the 2D Gaussians agree on the surface a ray hit (see TriangleTrust.h).
enum class SurfaceTrust : std::uint8_t
{
    Unknown,  // no hit, a Gaussian hit, or no Gaussians to compare with
    Trusted,  // mesh hit where both reconstructions agree
    Untrusted // mesh hit where they disagree: much more likely to be wrong
};

// One LiDAR sweep in the same layout for every mode (mesh, 2D Gaussian, hybrid).
// Every array has one entry per ray, ray index = beam * AzimuthCount + azimuth, so it reshapes to a range image.
struct LidarFrame
{
    std::uint32_t Frame = 0;
    std::uint32_t BeamCount = 0;
    std::uint32_t AzimuthCount = 0;
    Matrix4d WorldFromSensor{};

    std::vector<float> Ranges;                // metres, -1 when the ray missed
    std::vector<std::array<float, 3>> Points; // sensor frame (+x forward, +y left, +z up), 0 when missed
    std::vector<HitSource> Sources;
    std::vector<SurfaceTrust> Trust;
};

// hits and sources: one per ray, in sensor ray order. untrustedTriangles: from FindUntrustedTriangles, empty when
// there are no Gaussians to compare with.
[[nodiscard]]
LidarFrame MakeLidarFrame(const LidarSensor &sensor, const VelodynePose &pose, std::span<const rt::Hit> hits,
                          std::span<const HitSource> sources, std::span<const std::uint8_t> untrustedTriangles = {});

// Same source for every valid hit, for single-representation scans.
[[nodiscard]]
std::vector<HitSource> SourcesFor(std::span<const rt::Hit> hits, HitSource source);

// Binary .rlf file, read by python/raylidar_frame.py. Little-endian:
//   header (128 bytes): magic "RLFR" | version u32 = 1 | frame u32 | beam_count u32 | azimuth_count u32 |
//                       reserved u32[3] | world_from_sensor f64[12] (rows 0..2 of the 4x4 pose)
//   then N = beam_count * azimuth_count rays, one array after another:
//   range f32[N] | point f32[N][3] | source u8[N] (0 none, 1 mesh, 2 Gaussian) |
//   trust u8[N] (0 unknown, 1 trusted, 2 untrusted)
void WriteLidarFrame(const std::filesystem::path &path, const LidarFrame &frame);
} // namespace raylidar::simulation
