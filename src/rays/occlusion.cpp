#include <raylidar/rays/occlusion.h>

#include <raylidar/overlay/project.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace raylidar::rays
{
namespace
{
constexpr std::size_t kMaxBucketOccupancy = 8;

struct BucketEntry
{
    double depth = 0.0;
    std::size_t index = 0;
};

struct PointKey
{
    std::uint32_t x = 0;
    std::uint32_t y = 0;
    std::uint32_t z = 0;

    bool operator==(const PointKey &other) const noexcept
    {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct PointKeyHash
{
    std::size_t operator()(const PointKey &key) const noexcept
    {
        std::uint64_t hash = 1469598103934665603ULL;
        const auto mix = [&hash](const std::uint32_t value) {
            hash ^= static_cast<std::uint64_t>(value);
            hash *= 1099511628211ULL;
        };
        mix(key.x);
        mix(key.y);
        mix(key.z);
        return static_cast<std::size_t>(hash);
    }
};

std::uint32_t FloatBits(const float value)
{
    std::uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

PointKey KeyOf(const Eigen::Vector3f &point)
{
    return PointKey{FloatBits(point.x()), FloatBits(point.y()), FloatBits(point.z())};
}

std::uint64_t PixelKey(const int x, const int y, const int width)
{
    return static_cast<std::uint64_t>(y) * static_cast<std::uint64_t>(width) +
           static_cast<std::uint64_t>(x);
}
} // namespace

std::vector<std::uint8_t> ComputeOcclusionFlags(
    const std::vector<gt::GtPoint> &gtPoints, const std::vector<Ray> &rays, const int camera,
    const Eigen::Matrix4d &T_Ck_rect_V, const Eigen::Matrix3d &K_rect, const int width,
    const int height, const OcclusionOptions &options)
{
    if (camera != 0 && camera != 1)
    {
        throw std::invalid_argument("occlusion camera must be 0 (image_00) or 1 (image_01)");
    }
    if (width <= 0 || height <= 0)
    {
        throw std::invalid_argument("occlusion image dimensions must be positive");
    }
    if (options.neighborPx < 0)
    {
        throw std::invalid_argument("occlusion neighborPx must be non-negative");
    }
    if (!std::isfinite(options.sigmaLidarM) || options.sigmaLidarM < 0.0 ||
        !std::isfinite(options.sigmaMultiple) || options.sigmaMultiple < 0.0 ||
        !std::isfinite(options.depthThresholdM))
    {
        throw std::invalid_argument("occlusion thresholds must be finite and non-negative");
    }
    const double threshold = options.depthThresholdM > 0.0
                                 ? options.depthThresholdM
                                 : options.sigmaLidarM * options.sigmaMultiple;

    std::vector<std::uint8_t> flags(rays.size(), 0U);
    if (gtPoints.empty())
    {
        return flags;
    }

    // Project every GT point of the frame, including camera-255 points and
    // points assigned to the other camera (W5 all-visible z-buffer).
    Eigen::MatrixXd pointsV(static_cast<Eigen::Index>(gtPoints.size()), 3);
    for (std::size_t index = 0; index < gtPoints.size(); ++index)
    {
        pointsV.row(static_cast<Eigen::Index>(index)) = gtPoints[index].p_L.cast<double>();
    }
    const overlay::ProjectionResult projection =
        overlay::ProjectPoints(pointsV, T_Ck_rect_V, K_rect, width, height, 0.0, 1.0e9, 0, false);

    std::unordered_map<std::uint64_t, std::vector<BucketEntry>> buckets;
    buckets.reserve(gtPoints.size() / 2U + 1U);
    for (std::size_t index = 0; index < gtPoints.size(); ++index)
    {
        if (!projection.valid[index])
        {
            continue;
        }
        const double u = projection.uv[index].x();
        const double v = projection.uv[index].y();
        const double z = projection.z[index];
        if (!std::isfinite(u) || !std::isfinite(v) || !std::isfinite(z) || z <= 0.0)
        {
            continue;
        }
        const int px = static_cast<int>(std::floor(u));
        const int py = static_cast<int>(std::floor(v));
        if (px < 0 || px >= width || py < 0 || py >= height)
        {
            continue;
        }

        std::vector<BucketEntry> &entries = buckets[PixelKey(px, py, width)];
        const BucketEntry entry{z, index};
        const auto position =
            std::lower_bound(entries.begin(), entries.end(), entry,
                             [](const BucketEntry &left, const BucketEntry &right) {
                                 return left.depth < right.depth;
                             });
        entries.insert(position, entry);
        // Keep only the 8 nearest entries per pixel; farther points cannot make
        // a within-threshold point occluded sooner than the retained ones.
        if (entries.size() > kMaxBucketOccupancy)
        {
            entries.pop_back();
        }
    }

    // Index GT points by their exact float p_L so each ray can skip itself.
    std::unordered_map<PointKey, std::vector<std::size_t>, PointKeyHash> pointIndex;
    pointIndex.reserve(gtPoints.size());
    for (std::size_t index = 0; index < gtPoints.size(); ++index)
    {
        pointIndex[KeyOf(gtPoints[index].p_L)].push_back(index);
    }

    for (std::size_t rayIndex = 0; rayIndex < rays.size(); ++rayIndex)
    {
        const Ray &ray = rays[rayIndex];
        if (ray.camera != static_cast<std::uint8_t>(camera))
        {
            continue;
        }
        const double z = static_cast<double>(ray.p_C.z());
        if (!std::isfinite(z) || z <= 0.0)
        {
            continue;
        }
        const double u = K_rect(0, 0) * static_cast<double>(ray.p_C.x()) / z + K_rect(0, 2);
        const double v = K_rect(1, 1) * static_cast<double>(ray.p_C.y()) / z + K_rect(1, 2);
        if (!std::isfinite(u) || !std::isfinite(v))
        {
            continue;
        }
        const int px = static_cast<int>(std::floor(u));
        const int py = static_cast<int>(std::floor(v));
        if (px < 0 || px >= width || py < 0 || py >= height)
        {
            continue;
        }

        std::size_t ownIndex = std::numeric_limits<std::size_t>::max();
        const auto own = pointIndex.find(KeyOf(ray.p_L));
        if (own != pointIndex.end())
        {
            for (const std::size_t candidate : own->second)
            {
                if (candidate < gtPoints.size())
                {
                    ownIndex = candidate;
                    break;
                }
            }
        }

        bool occluded = false;
        const int x0 = std::max(0, px - options.neighborPx);
        const int x1 = std::min(width - 1, px + options.neighborPx);
        const int y0 = std::max(0, py - options.neighborPx);
        const int y1 = std::min(height - 1, py + options.neighborPx);
        for (int y = y0; y <= y1 && !occluded; ++y)
        {
            for (int x = x0; x <= x1 && !occluded; ++x)
            {
                const auto bucket = buckets.find(PixelKey(x, y, width));
                if (bucket == buckets.end())
                {
                    continue;
                }
                for (const BucketEntry &entry : bucket->second)
                {
                    if (entry.index == ownIndex)
                    {
                        continue;
                    }
                    if (z - entry.depth > threshold)
                    {
                        occluded = true;
                        break;
                    }
                }
            }
        }
        flags[rayIndex] = occluded ? 1U : 0U;
    }
    return flags;
}
} // namespace raylidar::rays
