#include "rt/ASBuilder.h"
#include "rt/MeshScene.h"
#include "simulation/LidarSensor.h"
#include "simulation/LidarSimulator.h"
#include "simulation/PointCloudIO.h"
#include "vulkan/CommandContext.h"
#include "vulkan/VulkanContext.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace
{
namespace sim = raylidar::simulation;

void Check(bool condition, const std::string &message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

// CPU-only: ray count, unit directions, origin from the pose translation.
void RunSensorTest()
{
    const sim::LidarSensor sensor(sim::LidarSensorDescription{});
    Check(sensor.RayCount() == 64u * 2048u, "Unexpected ray count.");

    sim::Transform pose = sim::kIdentityTransform;
    pose[0][3] = 1.0f;
    pose[1][3] = 2.0f;
    pose[2][3] = 3.0f;

    const auto rays = sensor.GenerateRays(pose);
    Check(rays.size() == sensor.RayCount(), "GenerateRays returned the wrong number of rays.");

    for (const auto &ray : rays)
    {
        const auto &d = ray.Direction;
        const float length = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);

        Check(std::abs(length - 1.0f) < 1.0e-5f, "Ray direction is not normalized.");
        Check(ray.Origin[0] == 1.0f && ray.Origin[1] == 2.0f && ray.Origin[2] == 3.0f, "Ray origin is wrong.");
    }

    // Beam 0, azimuth 0 points forward and down at the minimum elevation.
    const float expectedZ = std::sin(-24.8f * 3.14159265f / 180.0f);
    Check(rays[0].Direction[0] > 0.0f && std::abs(rays[0].Direction[2] - expectedZ) < 1.0e-5f,
          "First ray has the wrong direction.");

    std::cout << "LiDAR sensor test passed: " << rays.size() << " rays.\n";
}

// GPU: sensor at the origin, a 40 m x 40 m wall at x = 10. Compares every ray with a CPU reference.
void RunWallTest(const std::filesystem::path &shaderPath, const std::filesystem::path &plyPath)
{
    constexpr float wallX = 10.0f;
    constexpr float halfSize = 20.0f;

    const std::array<std::array<float, 3>, 4> vertices = {{
        {wallX, -halfSize, -halfSize},
        {wallX, halfSize, -halfSize},
        {wallX, halfSize, halfSize},
        {wallX, -halfSize, halfSize},
    }};
    const std::array<std::uint32_t, 6> indices = {0, 1, 2, 0, 2, 3};

    raylidar::vulkan::VulkanContext context;

    const raylidar::rt::MeshDescription mesh{vertices, indices};
    const raylidar::rt::MeshInstanceDescription instance{};

    raylidar::rt::MeshSceneDescription sceneDescription;
    sceneDescription.Meshes = std::span(&mesh, 1);
    sceneDescription.Instances = std::span(&instance, 1);

    raylidar::rt::MeshScene scene(context, sceneDescription);
    {
        raylidar::rt::ASBuilder builder(context);
        raylidar::vulkan::CommandContext buildCommands(context);
        scene.RecordBuild(buildCommands.Begin(), builder);
        buildCommands.End();
        buildCommands.SubmitAndWait();
        scene.ReleaseUploadResources();
    }

    const sim::LidarSensor sensor(sim::LidarSensorDescription{});
    const auto rays = sensor.GenerateRays(sim::kIdentityTransform);

    sim::LidarSimulator simulator(context, shaderPath, sensor.RayCount());
    const auto result = simulator.Scan(scene, rays);

    std::size_t expectedHits = 0;

    for (std::size_t index = 0; index < rays.size(); ++index)
    {
        const auto &d = rays[index].Direction;
        const auto &hit = result.Hits[index];

        bool expectHit = false;
        float expectedT = 0.0f;

        if (d[0] > 1.0e-6f)
        {
            expectedT = wallX / d[0];
            const float y = expectedT * d[1];
            const float z = expectedT * d[2];

            // Skip rays that graze the wall edge; either answer is acceptable there.
            if (std::abs(std::abs(y) - halfSize) < 1.0e-3f || std::abs(std::abs(z) - halfSize) < 1.0e-3f)
            {
                continue;
            }

            expectHit = std::abs(y) < halfSize && std::abs(z) < halfSize && expectedT <= sensor.Description().RangeMax;
        }

        Check((hit.Valid != 0) == expectHit, "Hit mismatch at ray " + std::to_string(index) + ".");

        if (expectHit)
        {
            ++expectedHits;
            Check(std::abs(hit.T - expectedT) < 1.0e-3f * expectedT,
                  "Range mismatch at ray " + std::to_string(index) + ".");
        }
    }

    Check(expectedHits > 0, "Wall test produced no hits.");

    std::cout << "LiDAR wall test passed: " << result.Points.size() << " / " << rays.size() << " rays hit, trace "
              << result.Timings.Trace << " ms.\n";

    if (!plyPath.empty())
    {
        sim::WritePointCloudPly(plyPath, result.Points);
        std::cout << "Wrote " << plyPath.string() << '\n';
    }
}
} // namespace

int main(int argc, char *argv[])
{
    try
    {
        const std::string_view mode = argc >= 2 ? argv[1] : "";

        if (mode == "sensor" && argc == 2)
        {
            RunSensorTest();
            return EXIT_SUCCESS;
        }

        if (mode == "wall" && (argc == 3 || argc == 4))
        {
            RunWallTest(argv[2], argc == 4 ? argv[3] : "");
            return EXIT_SUCCESS;
        }

        std::cerr << "Usage: raylidar_lidar_test sensor\n"
                  << "       raylidar_lidar_test wall <LidarTrace.spv> [output.ply]\n";
        return EXIT_FAILURE;
    }
    catch (const std::exception &exception)
    {
        std::cerr << "LiDAR test failed: " << exception.what() << '\n';
        return EXIT_FAILURE;
    }
}
