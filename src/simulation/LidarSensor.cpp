#include "LidarSensor.h"

#include <cmath>
#include <numbers>
#include <stdexcept>

namespace raylidar::simulation
{
LidarSensor::LidarSensor(const LidarSensorDescription &description) : m_description(description)
{
    if (description.BeamCount == 0 || description.AzimuthCount == 0)
    {
        throw std::invalid_argument("LidarSensor requires at least one beam and one azimuth step.");
    }

    if (description.RangeMin < 0.0f || description.RangeMax <= description.RangeMin)
    {
        throw std::invalid_argument("LidarSensor requires 0 <= RangeMin < RangeMax.");
    }

    constexpr float degreesToRadians = std::numbers::pi_v<float> / 180.0f;

    const float elevationMin = description.ElevationMinDegrees * degreesToRadians;
    const float elevationMax = description.ElevationMaxDegrees * degreesToRadians;

    m_localDirections.reserve(static_cast<std::size_t>(description.BeamCount) * description.AzimuthCount);

    for (std::uint32_t beam = 0; beam < description.BeamCount; ++beam)
    {
        const float t =
            description.BeamCount > 1 ? static_cast<float>(beam) / static_cast<float>(description.BeamCount - 1) : 0.0f;
        const float elevation = elevationMin + (elevationMax - elevationMin) * t;

        for (std::uint32_t step = 0; step < description.AzimuthCount; ++step)
        {
            const float azimuth = 2.0f * std::numbers::pi_v<float> * static_cast<float>(step) /
                                  static_cast<float>(description.AzimuthCount);

            m_localDirections.push_back({
                std::cos(elevation) * std::cos(azimuth),
                std::cos(elevation) * std::sin(azimuth),
                std::sin(elevation),
            });
        }
    }
}

std::vector<rt::Ray> LidarSensor::GenerateRays(const Transform &sensorToWorld) const
{
    const auto &m = sensorToWorld;

    std::vector<rt::Ray> rays(m_localDirections.size());

    for (std::size_t index = 0; index < rays.size(); ++index)
    {
        const auto &d = m_localDirections[index];

        std::array<float, 3> world = {
            m[0][0] * d[0] + m[0][1] * d[1] + m[0][2] * d[2],
            m[1][0] * d[0] + m[1][1] * d[1] + m[1][2] * d[2],
            m[2][0] * d[0] + m[2][1] * d[1] + m[2][2] * d[2],
        };

        const float length = std::sqrt(world[0] * world[0] + world[1] * world[1] + world[2] * world[2]);

        for (auto &component : world)
        {
            component /= length;
        }

        auto &ray = rays[index];
        ray.Origin = {m[0][3], m[1][3], m[2][3]};
        ray.TMin = m_description.RangeMin;
        ray.Direction = world;
        ray.TMax = m_description.RangeMax;
    }

    return rays;
}
} // namespace raylidar::simulation
