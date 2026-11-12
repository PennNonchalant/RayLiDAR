#include "SceneCoordinates.h"

#include <algorithm>
#include <cstddef>
#include <limits>

namespace raylidar::simulation
{
std::array<double, 3> BoundingBoxCenter(std::span<const std::array<double, 3>> positions)
{
    constexpr double largest = std::numeric_limits<double>::max();
    constexpr double lowest = std::numeric_limits<double>::lowest();

    std::array<double, 3> minimum{largest, largest, largest};
    std::array<double, 3> maximum{lowest, lowest, lowest};

    for (const auto &position : positions)
    {
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            minimum[axis] = std::min(minimum[axis], position[axis]);
            maximum[axis] = std::max(maximum[axis], position[axis]);
        }
    }

    return {
        0.5 * (minimum[0] + maximum[0]),
        0.5 * (minimum[1] + maximum[1]),
        0.5 * (minimum[2] + maximum[2]),
    };
}

std::vector<std::array<float, 3>> ToLocalFloat(std::span<const std::array<double, 3>> positions,
                                               const std::array<double, 3> &origin)
{
    std::vector<std::array<float, 3>> local;
    local.reserve(positions.size());

    for (const auto &position : positions)
    {
        local.push_back({
            static_cast<float>(position[0] - origin[0]),
            static_cast<float>(position[1] - origin[1]),
            static_cast<float>(position[2] - origin[2]),
        });
    }

    return local;
}
} // namespace raylidar::simulation