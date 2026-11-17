// Uses only the public header: proves the API works without any Vulkan or internal headers.
#include <raylidar/Simulator.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
constexpr double kWallX = 10.0;

void Check(bool condition, const std::string &message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

// 40 m x 40 m wall at x = 10, as a binary PLY with two triangles.
void WriteWallPly(const std::filesystem::path &path)
{
    const std::array<std::array<float, 3>, 4> vertices = {{
        {10.0f, -20.0f, -20.0f},
        {10.0f, 20.0f, -20.0f},
        {10.0f, 20.0f, 20.0f},
        {10.0f, -20.0f, 20.0f},
    }};
    const std::array<std::array<std::int32_t, 3>, 2> faces = {{{0, 1, 2}, {0, 2, 3}}};

    std::ofstream file(path, std::ios::binary);
    file << "ply\nformat binary_little_endian 1.0\n"
         << "element vertex 4\nproperty float x\nproperty float y\nproperty float z\n"
         << "element face 2\nproperty list uchar int vertex_indices\nend_header\n";

    file.write(reinterpret_cast<const char *>(vertices.data()), sizeof(vertices));

    for (const auto &face : faces)
    {
        const char count = 3;
        file.write(&count, 1);
        file.write(reinterpret_cast<const char *>(face.data()), sizeof(face));
    }
}

// The same wall as opaque 2D Gaussian splats facing the sensor, in the 2DGS point_cloud.ply layout.
void WriteSplatWallPly(const std::filesystem::path &path)
{
    constexpr int kSide = 79; // splats from -19.5 m to 19.5 m every 0.5 m
    const float logScale = std::log(0.3f);
    const float opacityLogit = std::log(0.99f / 0.01f);
    const float halfRoot = std::sqrt(0.5f);

    std::ofstream file(path, std::ios::binary);
    file << "ply\nformat binary_little_endian 1.0\nelement vertex " << kSide * kSide << '\n';
    for (const char *name : {"x", "y", "z", "opacity", "scale_0", "scale_1", "rot_0", "rot_1", "rot_2", "rot_3"})
    {
        file << "property float " << name << '\n';
    }
    file << "end_header\n";

    for (int row = 0; row < kSide; ++row)
    {
        for (int column = 0; column < kSide; ++column)
        {
            // Rotation of 90 degrees about +y turns the splat normal from +z to +x.
            const std::array<float, 10> record = {
                static_cast<float>(kWallX),
                -19.5f + 0.5f * static_cast<float>(column),
                -19.5f + 0.5f * static_cast<float>(row),
                opacityLogit,
                logScale,
                logScale,
                halfRoot,
                0.0f,
                halfRoot,
                0.0f,
            };
            file.write(reinterpret_cast<const char *>(record.data()), sizeof(record));
        }
    }
}

std::size_t CountSource(const raylidar::ScanOutput &output, raylidar::HitSource source)
{
    std::size_t count = 0;
    for (const auto value : output.Sources)
    {
        count += value == source ? 1 : 0;
    }
    return count;
}

void CheckOnWall(const raylidar::ScanOutput &output, const std::string &name)
{
    Check(output.Ranges.size() == output.Sources.size() && output.Ranges.size() == output.Trust.size() &&
              output.Ranges.size() == output.SensorPoints.size(),
          name + ": per-ray arrays have different sizes.");
    Check(!output.Points.empty(), name + ": no hits.");

    for (const auto &point : output.Points)
    {
        Check(std::abs(point[0] - kWallX) < 1.0e-2, name + ": hit point is not on the wall.");
    }
}
} // namespace

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: raylidar_api_test <LidarTrace.spv>\n";
        return EXIT_FAILURE;
    }

    try
    {
        const std::filesystem::path tracePath = argv[1];
        const auto temp = std::filesystem::temp_directory_path();
        const auto meshPath = temp / "raylidar_api_test_wall.ply";
        const auto splatPath = temp / "raylidar_api_test_splats.ply";
        const auto framePath = temp / "raylidar_api_test_frame.rlf";
        WriteWallPly(meshPath);
        WriteSplatWallPly(splatPath);

        const raylidar::Pose identity = {{
            {1.0, 0.0, 0.0, 0.0},
            {0.0, 1.0, 0.0, 0.0},
            {0.0, 0.0, 1.0, 0.0},
            {0.0, 0.0, 0.0, 1.0},
        }};

        // Mesh only, original constructor.
        raylidar::Simulator simulator(meshPath, tracePath);
        Check(simulator.RayCount() == 64u * 2048u, "Unexpected ray count.");

        const auto output = simulator.Scan(identity);
        Check(output.Ranges.size() == simulator.RayCount(), "Expected one range per ray.");
        CheckOnWall(output, "Mesh");

        // 46144 rays hit the wall with CPU-generated rays; GPU-generated rays may flip a ray that grazes an edge.
        const auto meshHits = output.Points.size();
        Check(meshHits + 16 >= 46144 && meshHits <= 46144 + 16, "Unexpected hit count: " + std::to_string(meshHits));
        Check(CountSource(output, raylidar::HitSource::Mesh) == meshHits, "Mesh hits must come from the mesh.");

        raylidar::SimulatorConfig config;
        config.MeshPath = meshPath;
        config.GaussianPath = splatPath;
        config.ShaderDirectory = tracePath.parent_path();

        // Gaussian only.
        config.Mode = raylidar::SceneMode::Gaussian;
        raylidar::Simulator gaussianSimulator(config);
        const auto gaussianOutput = gaussianSimulator.Scan(identity);
        CheckOnWall(gaussianOutput, "Gaussian");
        Check(CountSource(gaussianOutput, raylidar::HitSource::Gaussian) == gaussianOutput.Points.size(),
              "Gaussian hits must come from the Gaussians.");

        // Hybrid: the mesh answers every ray it hits, and agrees with the splats, so those hits are trusted.
        config.Mode = raylidar::SceneMode::Hybrid;
        raylidar::Simulator hybridSimulator(config);
        const auto hybridOutput = hybridSimulator.Scan(identity);
        CheckOnWall(hybridOutput, "Hybrid");
        const auto hybridMeshHits = CountSource(hybridOutput, raylidar::HitSource::Mesh);
        Check(hybridMeshHits + 16 >= meshHits && hybridMeshHits <= meshHits + 16, "Hybrid lost mesh hits.");

        std::size_t trusted = 0;
        for (const auto trust : hybridOutput.Trust)
        {
            trusted += trust == raylidar::SurfaceTrust::Trusted ? 1 : 0;
        }
        Check(trusted == hybridMeshHits, "Every mesh hit on the agreeing wall should be trusted.");

        // .rlf export: 128-byte header, then range f32, point f32x3, source u8 and trust u8 per ray.
        raylidar::Simulator::WriteFrame(framePath, hybridOutput, 7);
        const auto expectedBytes = 128 + hybridOutput.Ranges.size() * (4 + 12 + 1 + 1);
        Check(std::filesystem::file_size(framePath) == expectedBytes, "Unexpected .rlf size.");

        std::filesystem::remove(meshPath);
        std::filesystem::remove(splatPath);
        std::filesystem::remove(framePath);

        std::cout << "Public API test passed: mesh " << meshHits << " hits (" << output.Timings.Total
                  << " ms), Gaussian " << gaussianOutput.Points.size() << " hits (" << gaussianOutput.Timings.Total
                  << " ms), hybrid " << hybridOutput.Points.size() << " hits, " << trusted << " trusted.\n";
        return EXIT_SUCCESS;
    }
    catch (const std::exception &exception)
    {
        std::cerr << "Public API test failed: " << exception.what() << '\n';
        return EXIT_FAILURE;
    }
}
