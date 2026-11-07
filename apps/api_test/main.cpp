// Uses only the public header: proves the API works without any Vulkan or internal headers.
#include <raylidar/Simulator.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{
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
        const auto meshPath = std::filesystem::temp_directory_path() / "raylidar_api_test_wall.ply";
        WriteWallPly(meshPath);

        raylidar::Simulator simulator(meshPath, argv[1]);
        Check(simulator.RayCount() == 64u * 2048u, "Unexpected ray count.");

        const raylidar::Pose identity = {{
            {1.0, 0.0, 0.0, 0.0},
            {0.0, 1.0, 0.0, 0.0},
            {0.0, 0.0, 1.0, 0.0},
            {0.0, 0.0, 0.0, 1.0},
        }};

        const auto output = simulator.Scan(identity);

        Check(output.Ranges.size() == simulator.RayCount(), "Expected one range per ray.");
        Check(output.Points.size() == 46144, "Unexpected hit count: " + std::to_string(output.Points.size()));

        for (const auto &point : output.Points)
        {
            Check(std::abs(point[0] - 10.0) < 1.0e-3, "Hit point is not on the wall.");
        }

        std::filesystem::remove(meshPath);

        std::cout << "Public API test passed: " << output.Points.size() << " hits, " << output.Timings.Total
                  << " ms end-to-end.\n";
        return EXIT_SUCCESS;
    }
    catch (const std::exception &exception)
    {
        std::cerr << "Public API test failed: " << exception.what() << '\n';
        return EXIT_FAILURE;
    }
}
