#pragma once

#include "rt/TraceTypes.h"

#include <array>
#include <cstdint>
#include <vector>

namespace raylidar::simulation
{
// Row-major 3x4 sensor-to-world transform: [R | t].
using Transform = std::array<std::array<float, 4>, 3>;

inline constexpr Transform kIdentityTransform{{
    {1.0f, 0.0f, 0.0f, 0.0f},
    {0.0f, 1.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, 1.0f, 0.0f},
}};

// Defaults approximate the Velodyne HDL-64E used by KITTI-360.
// Sensor frame: +x forward, +y left, +z up.
struct LidarSensorDescription
{
    std::uint32_t BeamCount = 64;
    std::uint32_t AzimuthCount = 2048;

    float ElevationMinDegrees = -24.8f;
    float ElevationMaxDegrees = 2.0f;

    float RangeMin = 0.0f;
    float RangeMax = 120.0f;
};

class LidarSensor
{
  public:
    explicit LidarSensor(const LidarSensorDescription &description);

    [[nodiscard]]
    std::uint32_t RayCount() const noexcept
    {
        return static_cast<std::uint32_t>(m_localDirections.size());
    }

    [[nodiscard]]
    const LidarSensorDescription &Description() const noexcept
    {
        return m_description;
    }

    // Ray index = beam * AzimuthCount + azimuth.
    [[nodiscard]]
    std::vector<rt::Ray> GenerateRays(const Transform &sensorToWorld) const;

  private:
    LidarSensorDescription m_description;
    std::vector<std::array<float, 3>> m_localDirections;
};
} // namespace raylidar::simulation
