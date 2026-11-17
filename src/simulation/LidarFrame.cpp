#include "LidarFrame.h"

#include <fstream>
#include <stdexcept>

namespace raylidar::simulation
{
namespace
{
constexpr std::uint32_t kLidarFrameVersion = 1;

template <typename T> void WriteValue(std::ofstream &stream, const T &value)
{
    stream.write(reinterpret_cast<const char *>(&value), sizeof(T));
}

template <typename T> void WriteArray(std::ofstream &stream, const std::vector<T> &values)
{
    stream.write(reinterpret_cast<const char *>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(T)));
}
} // namespace

LidarFrame MakeLidarFrame(const LidarSensor &sensor, const VelodynePose &pose, std::span<const rt::Hit> hits,
                          std::span<const HitSource> sources, std::span<const std::uint8_t> untrustedTriangles)
{
    const auto directions = sensor.LocalDirections();
    if (hits.size() != directions.size() || sources.size() != directions.size())
    {
        throw std::invalid_argument("LidarFrame needs one hit and one source per sensor ray.");
    }

    LidarFrame frame;
    frame.Frame = pose.Frame;
    frame.BeamCount = sensor.Description().BeamCount;
    frame.AzimuthCount = sensor.Description().AzimuthCount;
    frame.WorldFromSensor = pose.WorldFromVelodyne;
    frame.Ranges.assign(directions.size(), -1.0f);
    frame.Points.assign(directions.size(), {0.0f, 0.0f, 0.0f});
    frame.Sources.assign(directions.size(), HitSource::None);
    frame.Trust.assign(directions.size(), SurfaceTrust::Unknown);

    for (std::size_t index = 0; index < directions.size(); ++index)
    {
        const auto &hit = hits[index];
        if (hit.Valid == 0)
        {
            continue;
        }

        // The pose only rotates the ray, so the hit distance along the world ray is the range in the sensor frame.
        const auto &direction = directions[index];
        frame.Ranges[index] = hit.T;
        frame.Points[index] = {hit.T * direction[0], hit.T * direction[1], hit.T * direction[2]};
        frame.Sources[index] = sources[index];

        if (sources[index] == HitSource::Mesh && hit.PrimitiveId < untrustedTriangles.size())
        {
            frame.Trust[index] =
                untrustedTriangles[hit.PrimitiveId] != 0 ? SurfaceTrust::Untrusted : SurfaceTrust::Trusted;
        }
    }

    return frame;
}

std::vector<HitSource> SourcesFor(std::span<const rt::Hit> hits, HitSource source)
{
    std::vector<HitSource> sources(hits.size(), HitSource::None);
    for (std::size_t index = 0; index < hits.size(); ++index)
    {
        if (hits[index].Valid != 0)
        {
            sources[index] = source;
        }
    }
    return sources;
}

void WriteLidarFrame(const std::filesystem::path &path, const LidarFrame &frame)
{
    const std::size_t rayCount = static_cast<std::size_t>(frame.BeamCount) * frame.AzimuthCount;
    if (frame.Ranges.size() != rayCount || frame.Points.size() != rayCount || frame.Sources.size() != rayCount ||
        frame.Trust.size() != rayCount)
    {
        throw std::invalid_argument("LidarFrame arrays must have one entry per ray.");
    }

    std::ofstream stream(path, std::ios::binary);
    if (!stream)
    {
        throw std::runtime_error("Failed to open LiDAR frame for writing: " + path.string());
    }

    stream.write("RLFR", 4);
    WriteValue(stream, kLidarFrameVersion);
    WriteValue(stream, frame.Frame);
    WriteValue(stream, frame.BeamCount);
    WriteValue(stream, frame.AzimuthCount);
    const std::array<std::uint32_t, 3> reserved{};
    WriteValue(stream, reserved);
    for (std::size_t row = 0; row < 3; ++row)
    {
        WriteValue(stream, frame.WorldFromSensor[row]);
    }

    WriteArray(stream, frame.Ranges);
    WriteArray(stream, frame.Points);
    WriteArray(stream, frame.Sources);
    WriteArray(stream, frame.Trust);

    if (!stream)
    {
        throw std::runtime_error("Failed to write LiDAR frame: " + path.string());
    }
}
} // namespace raylidar::simulation
