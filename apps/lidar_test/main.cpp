#include "rt/ASBuilder.h"
#include "rt/MeshScene.h"
#include "simulation/GpuRayGenerator.h"
#include "simulation/LidarSensor.h"
#include "simulation/LidarSimulator.h"
#include "simulation/PlyIO.h"
#include "simulation/TriangleTrust.h"
#include "vulkan/Buffer.h"
#include "vulkan/CommandContext.h"
#include "vulkan/Synchronization.h"
#include "vulkan/VulkanContext.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

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

// CPU-only: a triangle on the Gaussian surface is trusted, one 0.5 m off it is not, one far from any splat is not.
void RunTrustTest()
{
    const std::array<std::array<float, 3>, 9> positions{{
        {0.0f, 0.0f, 0.0f},
        {0.3f, 0.0f, 0.0f},
        {0.0f, 0.3f, 0.0f},
        {10.0f, 0.0f, 0.5f},
        {10.3f, 0.0f, 0.5f},
        {10.0f, 0.3f, 0.5f},
        {50.0f, 0.0f, 0.0f},
        {50.3f, 0.0f, 0.0f},
        {50.0f, 0.3f, 0.0f},
    }};
    const std::array<std::uint32_t, 9> indices{0, 1, 2, 3, 4, 5, 6, 7, 8};

    // Flat splats in the z = 0 plane under the first two triangles.
    std::array<raylidar::rt::Gaussian, 2> gaussians{};
    gaussians[0].Center = {0.1f, 0.1f, 0.0f};
    gaussians[1].Center = {10.1f, 0.1f, 0.0f};

    const auto trust = sim::FindUntrustedTriangles(positions, indices, gaussians);
    Check(trust.Untrusted.size() == 3, "Expected one flag per triangle.");
    Check(trust.Untrusted[0] == 0, "Triangle on the Gaussian surface should be trusted.");
    Check(trust.Untrusted[1] == 1, "Triangle 0.5 m off the Gaussian surface should be untrusted.");
    Check(trust.Untrusted[2] == 1, "Triangle with no splat nearby should be untrusted.");
    Check(trust.UntrustedCount == 2 && trust.NoNearbySplatCount == 1, "Unexpected untrusted counts.");
    std::cout << "Triangle trust test passed.\n";
}

void RunGaussianLoadTest()
{
    const auto path = std::filesystem::temp_directory_path() / "raylidar_gaussian_load_test.ply";
    {
        std::ofstream file(path, std::ios::binary);
        Check(static_cast<bool>(file), "Failed to create the temporary Gaussian PLY.");

        file << "ply\n"
             << "format binary_little_endian 1.0\n"
             << "element vertex 4\n"
             << "property float x\n"
             << "property float y\n"
             << "property float z\n"
             << "property float opacity\n"
             << "property float scale_0\n"
             << "property float scale_1\n"
             << "property float rot_0\n"
             << "property float rot_1\n"
             << "property float rot_2\n"
             << "property float rot_3\n"
             << "end_header\n";

        const auto logit = [](float value) { return std::log(value / (1.0f - value)); };
        const std::array<std::array<float, 10>, 4> records{{
            {0.0f, 0.0f, 0.0f, logit(0.1f), std::log(0.5f), std::log(0.5f), 1.0f, 0.0f, 0.0f, 0.0f},
            {1.0f, 0.0f, 0.0f, logit(0.9f), std::log(2.0f), std::log(0.5f), 1.0f, 0.0f, 0.0f, 0.0f},
            {2.0f, 0.0f, 0.0f, logit(0.25f), std::log(1.0f), std::log(0.5f), 1.0f, 0.0f, 0.0f, 0.0f},
            {3.0f, 0.0f, 0.0f, logit(0.9f), std::log(0.1f), std::log(0.1f), 1.0f, 0.0f, 0.0f, 0.0f},
        }};
        for (const auto &record : records)
        {
            file.write(reinterpret_cast<const char *>(record.data()),
                       static_cast<std::streamsize>(record.size() * sizeof(float)));
        }
    }

    sim::GaussianLoadStats stats;
    const auto pruned = sim::PlyIO::LoadGaussians(path, {0.0, 0.0, 0.0}, true, &stats);
    Check(pruned.size() == 2, "Prune should keep the boundary and the clean splat.");
    Check(stats.Input == 4 && stats.Kept == 2, "Unexpected input/kept counts.");
    Check(stats.RemovedOpacity == 1, "Expected one opacity rejection.");
    Check(stats.RemovedScale == 1, "Expected one scale rejection.");

    const auto all = sim::PlyIO::LoadGaussians(path, {0.0, 0.0, 0.0}, false);
    Check(all.size() == 4, "Prune-off should load every splat.");

    std::filesystem::remove(path);
    std::cout << "Gaussian load test passed.\n";
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
// With rayGenShaderPath, the same scan is repeated with rays generated on the GPU and checked the same way.
void RunWallTest(const std::filesystem::path &shaderPath, const std::filesystem::path &plyPath,
                 const std::filesystem::path &rayGenShaderPath = {})
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

    const auto checkWall = [&rays, &sensor](const sim::ScanResult &scan, std::string_view name) {
        std::size_t expectedHits = 0;

        for (std::size_t index = 0; index < rays.size(); ++index)
        {
            const auto &d = rays[index].Direction;
            const auto &hit = scan.Hits[index];

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

                expectHit =
                    std::abs(y) < halfSize && std::abs(z) < halfSize && expectedT <= sensor.Description().RangeMax;
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

        std::cout << "LiDAR wall test passed (" << name << "): " << scan.Points.size() << " / " << rays.size()
                  << " rays hit, trace " << scan.Timings.Trace << " ms.\n";
    };

    checkWall(result, "CPU rays");

    if (!rayGenShaderPath.empty())
    {
        simulator.EnableGpuRayGeneration(rayGenShaderPath, sensor);
        const auto gpuResult = simulator.Scan(scene, sensor, sim::kIdentityTransform);
        checkWall(gpuResult, "GPU rays");

        // Points are rebuilt on the CPU for GPU rays; they must land on the wall like the CPU-ray points.
        for (const auto &point : gpuResult.Points)
        {
            Check(std::abs(point[0] - wallX) < 1.0e-2f, "GPU-ray point is off the wall.");
        }
    }

    if (!plyPath.empty())
    {
        sim::PlyIO::WritePointCloud(plyPath, result.Points);
        std::cout << "Wrote " << plyPath.string() << '\n';
    }
}

// GPU: rays generated on the GPU must match LidarSensor's CPU rays for a non-trivial pose.
void RunRayGenTest(const std::filesystem::path &shaderPath)
{
    const sim::LidarSensorDescription description{};
    const sim::LidarSensor sensor(description);

    // Yaw of 30 degrees plus a translation far from the origin, like a real local pose.
    const float yaw = 30.0f * 3.14159265f / 180.0f;
    sim::Transform pose = sim::kIdentityTransform;
    pose[0][0] = std::cos(yaw);
    pose[0][1] = -std::sin(yaw);
    pose[1][0] = std::sin(yaw);
    pose[1][1] = std::cos(yaw);
    pose[0][3] = -49.5f;
    pose[1][3] = 5.1f;
    pose[2][3] = -5.4f;

    const auto expected = sensor.GenerateRays(pose);

    raylidar::vulkan::VulkanContext context;
    sim::GpuRayGenerator generator(context, shaderPath, sensor);

    const vk::DeviceSize rayBytes = expected.size() * sizeof(raylidar::rt::Ray);

    raylidar::vulkan::BufferDescription rayDescription;
    rayDescription.Size = rayBytes;
    rayDescription.Usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc;
    const raylidar::vulkan::Buffer rayBuffer(context, rayDescription);

    raylidar::vulkan::BufferDescription readbackDescription;
    readbackDescription.Size = rayBytes;
    readbackDescription.Usage = vk::BufferUsageFlagBits::eTransferDst;
    readbackDescription.MemoryUsage = raylidar::vulkan::BufferMemoryUsage::Readback;
    const raylidar::vulkan::Buffer readbackBuffer(context, readbackDescription);

    const std::array<std::uint32_t, 4> indices{sensor.RayCount() - 1, 0, 2048, 32769};
    raylidar::vulkan::BufferDescription indexDescription;
    indexDescription.Size = sizeof(indices);
    indexDescription.Usage = vk::BufferUsageFlagBits::eStorageBuffer;
    indexDescription.MemoryUsage = raylidar::vulkan::BufferMemoryUsage::Upload;
    raylidar::vulkan::Buffer indexBuffer(context, indexDescription);
    indexBuffer.Write(std::as_bytes(std::span(indices)));
    generator.SetBuffers(rayBuffer, indexBuffer);

    const vk::raii::QueryPool timestamps(context.Device(), vk::QueryPoolCreateInfo({}, vk::QueryType::eTimestamp, 2));
    raylidar::vulkan::CommandContext commands(context);
    float maxDirectionError = 0.0f;

    const auto checkRays = [&](std::uint32_t first, std::uint32_t count, bool indexed) {
        const auto &commandBuffer = commands.Begin();
        const vk::DeviceSize bytes = count * sizeof(raylidar::rt::Ray);
        commandBuffer.resetQueryPool(*timestamps, 0, 2);
        commandBuffer.writeTimestamp2(vk::PipelineStageFlagBits2::eComputeShader, *timestamps, 0);
        generator.Record(commandBuffer, description, pose, first, count, indexed);
        commandBuffer.writeTimestamp2(vk::PipelineStageFlagBits2::eComputeShader, *timestamps, 1);

        raylidar::vulkan::RecordMemoryBarrier(commandBuffer, vk::PipelineStageFlagBits2::eComputeShader,
                                              vk::AccessFlagBits2::eShaderWrite, vk::PipelineStageFlagBits2::eCopy,
                                              vk::AccessFlagBits2::eTransferRead);
        commandBuffer.copyBuffer(rayBuffer.Handle(), readbackBuffer.Handle(), vk::BufferCopy(0, 0, bytes));
        raylidar::vulkan::RecordMemoryBarrier(commandBuffer, vk::PipelineStageFlagBits2::eCopy,
                                              vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eHost,
                                              vk::AccessFlagBits2::eHostRead);
        commands.End();
        commands.SubmitAndWait();

        std::vector<raylidar::rt::Ray> actual(count);
        readbackBuffer.Read(std::as_writable_bytes(std::span(actual)));
        for (std::size_t index = 0; index < actual.size(); ++index)
        {
            const auto &a = actual[index];
            const auto &e = expected[indexed ? indices[index] : first + index];
            Check(a.Origin == e.Origin, "Origin mismatch at ray " + std::to_string(index) + ".");
            Check(a.TMin == e.TMin && a.TMax == e.TMax, "Range mismatch at ray " + std::to_string(index) + ".");
            for (std::size_t axis = 0; axis < 3; ++axis)
            {
                maxDirectionError = std::max(maxDirectionError, std::abs(a.Direction[axis] - e.Direction[axis]));
            }
        }
    };

    checkRays(32768, 67, false);
    checkRays(0, static_cast<std::uint32_t>(indices.size()), true);
    checkRays(0, sensor.RayCount(), false);
    Check(maxDirectionError < 1.0e-4f, "Direction mismatch: max error " + std::to_string(maxDirectionError) + ".");

    // CPU generation time for comparison.
    const auto cpuStart = std::chrono::steady_clock::now();
    const auto cpuRays = sensor.GenerateRays(pose);
    const double cpuMilliseconds =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cpuStart).count();

    const auto [queryResult, ticks] =
        timestamps.getResults<std::uint64_t>(0, 2, 2 * sizeof(std::uint64_t), sizeof(std::uint64_t),
                                             vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
    const double gpuMilliseconds = queryResult == vk::Result::eSuccess
                                       ? static_cast<double>(ticks[1] - ticks[0]) *
                                             context.PhysicalDeviceProperties().limits.timestampPeriod / 1.0e6
                                       : 0.0;

    std::cout << "GPU ray generation test passed: " << expected.size() << " rays, max direction error "
              << maxDirectionError << ". GPU " << gpuMilliseconds << " ms vs CPU " << cpuMilliseconds << " ms ("
              << cpuRays.size() << " rays).\n";
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

        if (mode == "trust" && argc == 2)
        {
            RunTrustTest();
            return EXIT_SUCCESS;
        }

        if (mode == "gaussian-load" && argc == 2)
        {
            RunGaussianLoadTest();
            return EXIT_SUCCESS;
        }

        if (mode == "raygen" && argc == 3)
        {
            RunRayGenTest(argv[2]);
            return EXIT_SUCCESS;
        }

        if (mode == "wall-gpu" && argc == 4)
        {
            RunWallTest(argv[2], "", argv[3]);
            return EXIT_SUCCESS;
        }

        if (mode == "wall" && (argc == 3 || argc == 4))
        {
            RunWallTest(argv[2], argc == 4 ? argv[3] : "");
            return EXIT_SUCCESS;
        }

        std::cerr << "Usage: raylidar_lidar_test sensor\n"
                  << "       raylidar_lidar_test trust\n"
                  << "       raylidar_lidar_test gaussian-load\n"
                  << "       raylidar_lidar_test wall <LidarTrace.spv> [output.ply]\n"
                  << "       raylidar_lidar_test raygen <LidarRayGen.spv>\n"
                  << "       raylidar_lidar_test wall-gpu <LidarTrace.spv> <LidarRayGen.spv>\n";
        return EXIT_FAILURE;
    }
    catch (const std::exception &exception)
    {
        std::cerr << "LiDAR test failed: " << exception.what() << '\n';
        return EXIT_FAILURE;
    }
}
