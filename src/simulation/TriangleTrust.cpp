#include "TriangleTrust.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace raylidar::simulation
{
namespace
{
using Vec3 = std::array<float, 3>;

constexpr std::int64_t kCellBias = 1 << 20;
constexpr std::uint64_t kCellMask = (1u << 21) - 1u;

// Uniform grid over splat centers. With cell size = search radius, every splat within the radius of a point lies in
// the 3x3x3 cells around it.
class SplatGrid
{
  public:
    SplatGrid(std::span<const rt::Gaussian> gaussians, float minOpacity, float cellSize)
        : m_gaussians(gaussians), m_inverseCellSize(1.0f / cellSize)
    {
        for (std::size_t index = 0; index < gaussians.size(); ++index)
        {
            if (gaussians[index].Opacity >= minOpacity)
            {
                m_cells[Key(Cell(gaussians[index].Center))].push_back(static_cast<std::uint32_t>(index));
            }
        }
    }

    // Index of the nearest splat center within maxDistance, or -1.
    [[nodiscard]]
    std::int64_t Nearest(const Vec3 &point, float maxDistance) const
    {
        const auto center = Cell(point);
        float bestSquared = maxDistance * maxDistance;
        std::int64_t best = -1;
        for (std::int64_t dx = -1; dx <= 1; ++dx)
        {
            for (std::int64_t dy = -1; dy <= 1; ++dy)
            {
                for (std::int64_t dz = -1; dz <= 1; ++dz)
                {
                    const auto found = m_cells.find(Key({center[0] + dx, center[1] + dy, center[2] + dz}));
                    if (found == m_cells.end())
                    {
                        continue;
                    }
                    for (const auto index : found->second)
                    {
                        const auto &splat = m_gaussians[index].Center;
                        const float x = splat[0] - point[0];
                        const float y = splat[1] - point[1];
                        const float z = splat[2] - point[2];
                        const float squared = x * x + y * y + z * z;
                        if (squared < bestSquared)
                        {
                            bestSquared = squared;
                            best = index;
                        }
                    }
                }
            }
        }
        return best;
    }

  private:
    [[nodiscard]]
    std::array<std::int64_t, 3> Cell(const Vec3 &point) const
    {
        return {static_cast<std::int64_t>(std::floor(point[0] * m_inverseCellSize)),
                static_cast<std::int64_t>(std::floor(point[1] * m_inverseCellSize)),
                static_cast<std::int64_t>(std::floor(point[2] * m_inverseCellSize))};
    }

    [[nodiscard]]
    static std::uint64_t Key(const std::array<std::int64_t, 3> &cell)
    {
        const auto pack = [](std::int64_t value) { return static_cast<std::uint64_t>(value + kCellBias) & kCellMask; };
        return pack(cell[0]) | (pack(cell[1]) << 21) | (pack(cell[2]) << 42);
    }

    std::span<const rt::Gaussian> m_gaussians;
    float m_inverseCellSize;
    std::unordered_map<std::uint64_t, std::vector<std::uint32_t>> m_cells;
};

Vec3 Cross(const Vec3 &a, const Vec3 &b)
{
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
} // namespace

TriangleTrustResult FindUntrustedTriangles(std::span<const std::array<float, 3>> positions,
                                           std::span<const std::uint32_t> indices,
                                           std::span<const rt::Gaussian> gaussians,
                                           const TriangleTrustSettings &settings)
{
    if (indices.size() % 3 != 0)
    {
        throw std::invalid_argument("Triangle index count must be a multiple of 3.");
    }
    if (settings.SearchRadius <= 0.0f)
    {
        throw std::invalid_argument("Search radius must be positive.");
    }

    const SplatGrid grid(gaussians, settings.MinOpacity, settings.SearchRadius);

    TriangleTrustResult result;
    const std::size_t triangleCount = indices.size() / 3;
    result.Untrusted.assign(triangleCount, 0);

    for (std::size_t triangle = 0; triangle < triangleCount; ++triangle)
    {
        Vec3 center{};
        for (std::size_t corner = 0; corner < 3; ++corner)
        {
            const auto &vertex = positions[indices[3 * triangle + corner]];
            for (std::size_t axis = 0; axis < 3; ++axis)
            {
                center[axis] += vertex[axis] / 3.0f;
            }
        }

        const auto nearest = grid.Nearest(center, settings.SearchRadius);
        if (nearest < 0)
        {
            result.Untrusted[triangle] = 1;
            ++result.UntrustedCount;
            ++result.NoNearbySplatCount;
            continue;
        }

        // A 2D Gaussian is a flat disk; its normal is perpendicular to both axes.
        const auto &splat = gaussians[static_cast<std::size_t>(nearest)];
        const Vec3 normal = Cross(splat.AxisU, splat.AxisV);
        const float length = std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
        float planeDistance = std::numeric_limits<float>::infinity();
        if (length > 0.0f)
        {
            planeDistance =
                std::abs((center[0] - splat.Center[0]) * normal[0] + (center[1] - splat.Center[1]) * normal[1] +
                         (center[2] - splat.Center[2]) * normal[2]) /
                length;
        }

        if (planeDistance > settings.MaxPlaneDistance)
        {
            result.Untrusted[triangle] = 1;
            ++result.UntrustedCount;
        }
    }

    return result;
}
} // namespace raylidar::simulation
