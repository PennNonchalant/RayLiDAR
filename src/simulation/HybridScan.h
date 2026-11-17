#pragma once

#include "LidarFrame.h"
#include "LidarSimulator.h"
#include "rt/TraceTypes.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace raylidar::rt
{
class GaussianScene;
class MeshScene;
} // namespace raylidar::rt

namespace raylidar::simulation
{
struct HybridScanResult
{
    // Hits, points and timings combined over both passes, in input ray order.
    ScanResult Result;

    std::vector<HitSource> Sources;

    std::size_t MeshHits = 0;
    std::size_t GaussianHits = 0;
    std::size_t RaysSentToGaussian = 0;

    double MeshMilliseconds = 0.0;     // CPU wall time of the mesh pass
    double RoutingMilliseconds = 0.0;  // CPU: pick rays for the Gaussian pass and merge
    double GaussianMilliseconds = 0.0; // CPU wall time of the Gaussian pass
};

// Traces Gaussians in chunks of rays, one GPU submission each. A single large Gaussian dispatch can run for seconds,
// and Windows resets the GPU (device lost) when one submission takes longer than about 2 s.
[[nodiscard]]
ScanResult ScanGaussiansInChunks(LidarSimulator &simulator, const rt::GaussianScene &scene,
                                 std::span<const rt::Ray> rays, float alphaThreshold, std::size_t raysPerChunk = 32768);

// Ground-truth-free hybrid routing.
// Every ray is traced against the mesh first: the mesh is the most accurate where it exists and is ~1000x cheaper.
// A ray goes to the 2D Gaussians only when the mesh gives no answer (a hole), or when it hits a triangle flagged as
// unreliable. The flags are computed offline from the reconstruction itself, never from LiDAR ground truth.
class HybridScanner
{
  public:
    HybridScanner(LidarSimulator &meshSimulator, LidarSimulator &gaussianSimulator);

    // untrustedTriangles: one entry per mesh triangle, non-zero = route to Gaussians. Empty = trust every triangle.
    // Measured on KITTI-360 4615-4674: routing untrusted triangles (TriangleTrust.h) made accuracy worse, because the
    // Gaussians are wrong in the same places as the mesh. The default routes only mesh misses.
    [[nodiscard]]
    HybridScanResult Scan(const rt::MeshScene &meshScene, const rt::GaussianScene &gaussianScene,
                          std::span<const rt::Ray> rays, float alphaThreshold,
                          std::span<const std::uint8_t> untrustedTriangles = {});

  private:
    LidarSimulator &m_meshSimulator;
    LidarSimulator &m_gaussianSimulator;
};
} // namespace raylidar::simulation
