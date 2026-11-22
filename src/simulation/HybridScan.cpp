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

bool IsUntrusted(const rt::Hit &hit, std::span<const std::uint8_t> untrustedTriangles)
{
    return !untrustedTriangles.empty() && hit.PrimitiveId < untrustedTriangles.size() &&
           untrustedTriangles[hit.PrimitiveId] != 0;
}

template <typename ScanChunk>
ScanResult ScanInChunks(std::size_t rayCount, std::size_t raysPerChunk, const ScanChunk &scanChunk)
{
    if (rayCount == 0)
    {
        return {};
    }
    if (raysPerChunk == 0 || raysPerChunk >= rayCount)
    {
        return scanChunk(0, rayCount);
    }

    const auto scanStart = Clock::now();
    ScanResult combined;
    combined.Hits.reserve(rayCount);
    combined.Points.reserve(rayCount);

    for (std::size_t first = 0; first < rayCount; first += raysPerChunk)
    {
        const auto count = std::min(raysPerChunk, rayCount - first);
        auto chunk = scanChunk(first, count);

        combined.Hits.insert(combined.Hits.end(), chunk.Hits.begin(), chunk.Hits.end());
        combined.Points.insert(combined.Points.end(), chunk.Points.begin(), chunk.Points.end());
        combined.Timings += chunk.Timings;
    }

    combined.Timings.Total = MillisecondsSince(scanStart);
    return combined;
}

template <typename ScanMesh, typename SelectRays>
HybridScanResult ScanHybrid(LidarSimulator &gaussianSimulator, const rt::GaussianScene &gaussianScene,
                            std::size_t rayCount, float alphaThreshold, std::size_t gaussianChunkSize,
                            std::span<const std::uint8_t> untrustedTriangles, const ScanMesh &scanMesh,
                            const SelectRays &selectRays)
{
    const auto scanStart = Clock::now();
    HybridScanResult output;

    const auto meshStart = Clock::now();
    auto meshResult = scanMesh();
    output.MeshMilliseconds = MillisecondsSince(meshStart);
    output.Result.Timings += meshResult.Timings;

    auto routeStart = Clock::now();
    std::vector<std::uint32_t> gaussianIndices;

    for (std::size_t index = 0; index < rayCount; ++index)
    {
        const auto &hit = meshResult.Hits[index];

        if (hit.Valid == 0 || IsUntrusted(hit, untrustedTriangles))
        {
            gaussianIndices.push_back(static_cast<std::uint32_t>(index));
        }
    }

    output.RaysSentToGaussian = gaussianIndices.size();

    ScanResult gaussianResult;

    if (!gaussianIndices.empty())
    {
        const auto gaussianRays = selectRays(std::span<const std::uint32_t>(gaussianIndices));
        const auto gaussianStart = Clock::now();
        output.Result.Timings.Routing = MillisecondsSince(routeStart);
        gaussianResult =
            ScanGaussians(gaussianSimulator, gaussianScene, gaussianRays, alphaThreshold, gaussianChunkSize);
        output.GaussianMilliseconds = MillisecondsSince(gaussianStart);
        output.Result.Timings += gaussianResult.Timings;
    }
    else
    {
        output.Result.Timings.Routing = MillisecondsSince(routeStart);
    }

    routeStart = Clock::now();
    auto &hits = output.Result.Hits;
    hits = std::move(meshResult.Hits);
    auto &points = output.Result.Points;
    points.reserve(rayCount);
    output.Sources.resize(rayCount, HitSource::None);
    auto meshPoint = meshResult.Points.begin();
    auto gaussianPoint = gaussianResult.Points.begin();
    std::size_t gaussianSlot = 0;

    for (std::size_t index = 0; index < rayCount; ++index)
    {
        const bool meshHit = hits[index].Valid != 0;
        if (gaussianSlot < gaussianIndices.size() && gaussianIndices[gaussianSlot] == index)
        {
            const auto &hit = gaussianResult.Hits[gaussianSlot++];
            if (hit.Valid != 0)
            {
                hits[index] = hit;
                points.push_back(*gaussianPoint++);
                output.Sources[index] = HitSource::Gaussian;
                ++output.GaussianHits;
                if (meshHit)
                {
                    ++meshPoint;
                }
                continue;
            }
        }

        if (meshHit)
        {
            points.push_back(*meshPoint++);
            output.Sources[index] = HitSource::Mesh;
            ++output.MeshHits;
        }
    }

    output.Result.Timings.Routing += MillisecondsSince(routeStart);
    output.Result.Timings.Total = MillisecondsSince(scanStart);

    return output;
}
} // namespace

ScanResult ScanGaussians(LidarSimulator &simulator, const rt::GaussianScene &scene, std::span<const rt::Ray> rays,
                         float alphaThreshold, std::size_t raysPerChunk)
{
    return ScanInChunks(rays.size(), raysPerChunk, [&](std::size_t first, std::size_t count) {
        return simulator.Scan(scene, rays.subspan(first, count), alphaThreshold);
    });
}

ScanResult ScanGaussians(LidarSimulator &simulator, const rt::GaussianScene &scene, const SensorRays &rays,
                         float alphaThreshold, std::size_t raysPerChunk)
{
    return ScanInChunks(rays.Count(), raysPerChunk, [&](std::size_t first, std::size_t count) {
        return simulator.Scan(scene, rays, alphaThreshold, first, count);
    });
}

HybridScanner::HybridScanner(LidarSimulator &meshSimulator, LidarSimulator &gaussianSimulator,
                             std::size_t gaussianChunkSize)
    : m_meshSimulator(meshSimulator), m_gaussianSimulator(gaussianSimulator), m_gaussianChunkSize(gaussianChunkSize)
{
}

HybridScanResult HybridScanner::Scan(const rt::MeshScene &meshScene, const rt::GaussianScene &gaussianScene,
                                     std::span<const rt::Ray> rays, float alphaThreshold,
                                     std::span<const std::uint8_t> untrustedTriangles)
{
    return ScanHybrid(
        m_gaussianSimulator, gaussianScene, rays.size(), alphaThreshold, m_gaussianChunkSize, untrustedTriangles,
        [&] { return m_meshSimulator.Scan(meshScene, rays); },
        [rays](std::span<const std::uint32_t> indices) {
            std::vector<rt::Ray> selected;
            selected.reserve(indices.size());
            for (const auto index : indices)
            {
                selected.push_back(rays[index]);
            }
            return selected;
        });
}

HybridScanResult HybridScanner::Scan(const rt::MeshScene &meshScene, const rt::GaussianScene &gaussianScene,
                                     const LidarSensor &sensor, const Transform &sensorToWorld, float alphaThreshold,
                                     std::span<const std::uint8_t> untrustedTriangles)
{
    return ScanHybrid(
        m_gaussianSimulator, gaussianScene, sensor.RayCount(), alphaThreshold, m_gaussianChunkSize, untrustedTriangles,
        [&] { return m_meshSimulator.Scan(meshScene, sensor, sensorToWorld); },
        [&](std::span<const std::uint32_t> indices) { return SensorRays{sensor, sensorToWorld, indices}; });
}

} // namespace raylidar::simulation
