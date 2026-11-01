#include "fixture.h"

#include <raylidar/rays/rays.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace raylidar::test
{
namespace
{
std::vector<Eigen::Vector3d> MakeUniformPoints(const std::size_t count, const std::uint64_t seed,
                                               const double half_extent)
{
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-half_extent, half_extent);
    std::vector<Eigen::Vector3d> points;
    points.reserve(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        points.emplace_back(dist(rng), dist(rng), dist(rng));
    }
    return points;
}
} // namespace

std::vector<Eigen::Vector3d> MakeDeterministicPoints(const std::size_t count,
                                                     const std::uint64_t seed)
{
    return MakeUniformPoints(count, seed, 10.0);
}

std::vector<Eigen::Vector3d> MakeDeterministicQueries(const std::size_t count,
                                                      const std::uint64_t seed)
{
    return MakeUniformPoints(count, seed, 12.0);
}

std::vector<Eigen::Vector3d> MakeWorldScalePoints(const std::size_t count,
                                                  const std::uint64_t seed)
{
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-5.0, 5.0);
    std::uniform_real_distribution<double> micro(-1.0e-3, 1.0e-3);
    std::vector<Eigen::Vector3d> points;
    points.reserve(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        // Every third point carries a sub-millimetre perturbation, so a float32
        // backend would collapse distinct points and return the wrong distance.
        const double dx = (i % 3U == 0U) ? micro(rng) : dist(rng);
        const double dy = (i % 3U == 1U) ? micro(rng) : dist(rng);
        const double dz = (i % 3U == 2U) ? micro(rng) : dist(rng);
        points.emplace_back(4000.0 + dx, 500.0 + dy, 100.0 + dz);
    }
    return points;
}

std::size_t BruteNearestIndex(const std::vector<Eigen::Vector3d> &points,
                              const Eigen::Vector3d &query)
{
    std::size_t best = 0;
    double bestDistance = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < points.size(); ++i)
    {
        const double distance = (points[i] - query).norm();
        if (distance < bestDistance)
        {
            bestDistance = distance;
            best = i;
        }
    }
    return best;
}

double BruteNearestDistance(const std::vector<Eigen::Vector3d> &points,
                            const Eigen::Vector3d &query)
{
    double bestDistance = std::numeric_limits<double>::infinity();
    for (const auto &point : points)
    {
        bestDistance = std::min(bestDistance, (point - query).norm());
    }
    return bestDistance;
}

std::vector<std::pair<std::size_t, double>>
BruteRadius(const std::vector<Eigen::Vector3d> &points, const Eigen::Vector3d &query,
            const double radius)
{
    std::vector<std::pair<std::size_t, double>> matches;
    for (std::size_t i = 0; i < points.size(); ++i)
    {
        const double distance = (points[i] - query).norm();
        if (distance <= radius)
        {
            matches.emplace_back(i, distance);
        }
    }
    std::sort(matches.begin(), matches.end(), [](const auto &lhs, const auto &rhs) {
        if (lhs.second != rhs.second)
        {
            return lhs.second < rhs.second;
        }
        return lhs.first < rhs.first;
    });
    return matches;
}

raylidar::eval::RaySample MakeSample(const std::uint64_t ray_id, const std::uint32_t frame,
                                     const std::uint16_t camera, const double t_gt,
                                     const double t_pred, const Eigen::Vector3d &origin,
                                     const Eigen::Vector3d &dir)
{
    using raylidar::rays::BucketOf;
    using raylidar::rays::kRayMaxRangeM;
    using raylidar::rays::kRayTMinM;

    raylidar::eval::RaySample sample;
    sample.ray_id = ray_id;
    sample.frame = frame;
    sample.camera = camera;
    sample.bucket = BucketOf(t_gt);
    sample.t_gt = t_gt;
    sample.t_min = kRayTMinM;
    sample.t_max = kRayMaxRangeM;
    sample.t_pred = t_pred;
    sample.origin_W = origin;
    sample.dir_W = dir.normalized();
    sample.occluded = false;
    return sample;
}

std::vector<raylidar::eval::RaySample> MakeIdentitySamples(const std::vector<double> &ranges,
                                                           const std::uint32_t frame,
                                                           const std::uint16_t camera)
{
    std::vector<raylidar::eval::RaySample> samples;
    samples.reserve(ranges.size());
    std::uint64_t rayId = 0;
    for (const double range : ranges)
    {
        samples.push_back(MakeSample(rayId++, frame, camera, range, range));
    }
    return samples;
}
} // namespace raylidar::test
