#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace raylidar::rt
{
struct Ray
{
    std::array<float, 3> Origin{};
    float TMin = 0.0f;

    std::array<float, 3> Direction{0.0f, 0.0f, 1.0f};
    float TMax = std::numeric_limits<float>::max();
};

struct Hit
{
    float T = 0.0f;
    std::uint32_t Valid = 0;

    std::uint32_t InstanceId = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t PrimitiveId = std::numeric_limits<std::uint32_t>::max();

    std::array<float, 2> Barycentrics{};
};
} // namespace raylidar::rt
