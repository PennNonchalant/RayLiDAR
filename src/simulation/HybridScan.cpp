#include "HybridScan.h"

#include <algorithm>
#include <chrono>

namespace raylidar::simulation
{
namespace
{
using Clock = std::chrono::steady_clock;

double MillisecondsSince(Clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void AddTimings(ScanTimings &sum, const ScanTimings &part)
{
    sum.Upload += part.Upload;
    sum.Trace += part.Trace;
    sum.CopyHits += part.CopyHits;
    sum.SubmitWait += part.SubmitWait;
    sum.Readback += part.Readback;
}

bool IsUntrusted(const rt::Hit &hit, std::span<const std::uint8_t> untrustedTriangles)
{
    return !untrustedTriangles.empty() && hit.PrimitiveId < untrustedTriangles.size() &&
           untrustedTriangles[hit.PrimitiveId] != 0;
}
} // namespace

ScanResult ScanGaussiansInChunks(LidarSimulator &simulator, const rt::GaussianScene &scene,
                                 std::span<const rt::Ray> rays, float alphaThreshold, std::size_t raysPerChunk)
{
    const auto scanStart = Clock::now();
    ScanResult combined;
    combined.Hits.reserve(rays.size());

    for (std::size_t first = 0; first < rays.size(); first += raysPerChunk)
    {
        const auto count = std::min(raysPerChunk, rays.size() - first);
        auto chunk = simulator.Scan(scene, rays.subspan(first, count), alphaThreshold);

        combined.Hits.insert(combined.Hits.end(), chunk.Hits.begin(), chunk.Hits.end());
        combined.Points.insert(combined.Points.end(), chunk.Points.begin(), chunk.Points.end());
        AddTimings(combined.Timings, chunk.Timings);
    }

    combined.Timings.Total = MillisecondsSince(scanStart);
    return combined;
}

HybridScanner::HybridScanner(LidarSimulator &meshSimulator, LidarSimulator &gaussianSimulator)
    : m_meshSimulator(meshSimulator), m_gaussianSimulator(gaussianSimulator)
{
}

HybridScanResult HybridScanner::Scan(const rt::MeshScene &meshScene, const rt::GaussianScene &gaussianScene,
                                     std::span<const rt::Ray> rays, float alphaThreshold,
                                     std::span<const std::uint8_t> untrustedTriangles)
{
    const auto scanStart = Clock::now();
    HybridScanResult output;

    // Pass 1: every ray against the mesh.
    const auto meshStart = Clock::now();
    auto meshResult = m_meshSimulator.Scan(meshScene, rays);
    output.MeshMilliseconds = MillisecondsSince(meshStart);
    AddTimings(output.Result.Timings, meshResult.Timings);

    // Route: holes and untrusted triangles go to the Gaussians.
    auto routeStart = Clock::now();
    std::vector<std::uint32_t> gaussianIndices;
    std::vector<rt::Ray> gaussianRays;

    for (std::size_t index = 0; index < rays.size(); ++index)
    {
        const auto &hit = meshResult.Hits[index];

        if (hit.Valid == 0 || IsUntrusted(hit, untrustedTriangles))
        {
            gaussianIndices.push_back(static_cast<std::uint32_t>(index));
            gaussianRays.push_back(rays[index]);
        }
    }

    output.RaysSentToGaussian = gaussianRays.size();
    output.RoutingMilliseconds = MillisecondsSince(routeStart);

    // Pass 2: only the routed rays against the Gaussians.
    ScanResult gaussianResult;

    if (!gaussianRays.empty())
    {
        const auto gaussianStart = Clock::now();
        gaussianResult = ScanGaussiansInChunks(m_gaussianSimulator, gaussianScene, gaussianRays, alphaThreshold);
        output.GaussianMilliseconds = MillisecondsSince(gaussianStart);
        AddTimings(output.Result.Timings, gaussianResult.Timings);
    }

    // Merge in input order.
    routeStart = Clock::now();
    auto &hits = output.Result.Hits;
    hits = std::move(meshResult.Hits);
    output.Sources.assign(rays.size(), HitSource::None);

    for (std::size_t index = 0; index < hits.size(); ++index)
    {
        if (hits[index].Valid != 0)
        {
            output.Sources[index] = HitSource::Mesh;
        }
    }

    for (std::size_t slot = 0; slot < gaussianIndices.size(); ++slot)
    {
        const auto index = gaussianIndices[slot];
        const auto &gaussianHit = gaussianResult.Hits[slot];

        // No Gaussian answer: an untrusted mesh hit is still better than nothing, so it stays.
        if (gaussianHit.Valid != 0)
        {
            hits[index] = gaussianHit;
            output.Sources[index] = HitSource::Gaussian;
        }
    }

    auto &points = output.Result.Points;
    points.reserve(rays.size());

    for (std::size_t index = 0; index < hits.size(); ++index)
    {
        const auto &hit = hits[index];

        if (hit.Valid == 0)
        {
            continue;
        }

        output.MeshHits += output.Sources[index] == HitSource::Mesh ? 1 : 0;
        output.GaussianHits += output.Sources[index] == HitSource::Gaussian ? 1 : 0;

        const auto &ray = rays[index];
        points.push_back({
            ray.Origin[0] + hit.T * ray.Direction[0],
            ray.Origin[1] + hit.T * ray.Direction[1],
            ray.Origin[2] + hit.T * ray.Direction[2],
        });
    }

    output.RoutingMilliseconds += MillisecondsSince(routeStart);
    output.Result.Timings.Readback += output.RoutingMilliseconds;
    output.Result.Timings.Total = MillisecondsSince(scanStart);

    return output;
}
} // namespace raylidar::simulation
