// Tests the public API through the public header only: no Vulkan or internal headers.
//
// Scene: a 40 m x 40 m wall at x = 10, both as a triangle mesh and as opaque 2D Gaussian splats, scanned from the
// origin. Usage: raylidar_api_test <LidarTrace.spv> (the other shaders are looked up in the same folder).
#include <raylidar/Simulator.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
constexpr double kWallX = 10.0;

// Rays that hit the wall with CPU-generated rays. GPU-generated rays may flip a few rays that graze an edge.
constexpr std::size_t kWallHits = 46144;
constexpr std::size_t kEdgeTolerance = 16;

constexpr raylidar::Pose kIdentity = {{
    {1.0, 0.0, 0.0, 0.0},
    {0.0, 1.0, 0.0, 0.0},
    {0.0, 0.0, 1.0, 0.0},
    {0.0, 0.0, 0.0, 1.0},
}};

void Check(bool condition, const std::string &message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

bool Near(std::size_t value, std::size_t expected, std::size_t tolerance)
{
    return value + tolerance >= expected && value <= expected + tolerance;
}

template <typename Exception> void CheckThrows(const std::function<void()> &action, const std::string &what)
{
    try
    {
        action();
    }
    catch (const Exception &)
    {
        return;
    }
    catch (const std::exception &exception)
    {
        throw std::runtime_error(what + ": threw the wrong exception type: " + exception.what());
    }
    throw std::runtime_error(what + ": did not throw.");
}

// Removes its folder on destruction, so a failed test leaves nothing behind.
class TempDirectory
{
  public:
    TempDirectory()
        : m_path(std::filesystem::temp_directory_path() /
                 ("raylidar_api_test_" + std::to_string(std::random_device{}())))
    {
        std::filesystem::create_directories(m_path);
    }

    ~TempDirectory()
    {
        std::error_code error;
        std::filesystem::remove_all(m_path, error);
    }

    TempDirectory(const TempDirectory &) = delete;
    TempDirectory &operator=(const TempDirectory &) = delete;
    TempDirectory(TempDirectory &&) = delete;
    TempDirectory &operator=(TempDirectory &&) = delete;

    [[nodiscard]]
    const std::filesystem::path &Path() const noexcept
    {
        return m_path;
    }

  private:
    std::filesystem::path m_path;
};

// The wall as a binary PLY with two triangles.
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

struct Fixture
{
    std::filesystem::path TraceShader;
    std::filesystem::path MeshPath;
    std::filesystem::path SplatPath;
    std::filesystem::path Directory;

    [[nodiscard]]
    raylidar::SimulatorConfig Config(raylidar::SceneMode mode) const
    {
        raylidar::SimulatorConfig config;
        config.Mode = mode;
        config.MeshPath = MeshPath;
        config.GaussianPath = SplatPath;
        config.ShaderDirectory = TraceShader.parent_path();
        return config;
    }
};

std::size_t Count(const std::vector<raylidar::HitSource> &values, raylidar::HitSource value)
{
    std::size_t count = 0;
    for (const auto entry : values)
    {
        count += entry == value ? 1 : 0;
    }
    return count;
}

std::size_t Count(const std::vector<raylidar::SurfaceTrust> &values, raylidar::SurfaceTrust value)
{
    std::size_t count = 0;
    for (const auto entry : values)
    {
        count += entry == value ? 1 : 0;
    }
    return count;
}

// Shape and geometry every mode must satisfy on the wall scene.
void CheckWallScan(const raylidar::Simulator &simulator, const raylidar::ScanOutput &output)
{
    const std::size_t rays = simulator.RayCount();
    Check(output.BeamCount * output.AzimuthCount == rays, "BeamCount * AzimuthCount must equal RayCount().");
    Check(output.Ranges.size() == rays && output.SensorPoints.size() == rays && output.Sources.size() == rays &&
              output.Trust.size() == rays,
          "Every per-ray array needs one entry per ray.");
    Check(output.Points.size() == rays - Count(output.Sources, raylidar::HitSource::None), "One world point per hit.");

    for (const auto &point : output.Points)
    {
        Check(std::abs(point[0] - kWallX) < 1.0e-2, "Hit point is not on the wall.");
    }
    for (std::size_t index = 0; index < rays; ++index)
    {
        const bool hit = output.Sources[index] != raylidar::HitSource::None;
        Check(hit == (output.Ranges[index] >= 0.0f), "A ray has a range exactly when it hit.");
    }

    const auto &timings = output.Timings;
    Check(timings.Total > 0.0 && timings.Total >= timings.Trace, "Total must cover the whole scan.");
}

// ---------------------------------------------------------------------------------------------------------------------

void MeshModeScansWall(const Fixture &fixture)
{
    raylidar::Simulator simulator(fixture.MeshPath, fixture.TraceShader);
    Check(simulator.Mode() == raylidar::SceneMode::Mesh, "The legacy constructor must use Mesh mode.");
    Check(!simulator.UsesGpuRayGeneration(), "The legacy constructor must keep CPU ray generation.");
    Check(simulator.RayCount() == 64u * 2048u, "Unexpected ray count.");

    const auto output = simulator.Scan(kIdentity);
    CheckWallScan(simulator, output);
    const auto hits = output.Points.size();
    Check(Near(hits, kWallHits, kEdgeTolerance), "Unexpected hit count: " + std::to_string(hits));
    Check(Count(output.Sources, raylidar::HitSource::Mesh) == hits, "Mesh hits must come from the mesh.");
    Check(Count(output.Trust, raylidar::SurfaceTrust::Unknown) == output.Trust.size(),
          "Without Gaussians every ray must be Unknown.");
}

void CpuAndGpuRayGenerationAgree(const Fixture &fixture)
{
    auto config = fixture.Config(raylidar::SceneMode::Mesh);
    config.GaussianPath.clear();

    config.PreferGpuRayGeneration = false;
    raylidar::Simulator cpu(config);
    Check(!cpu.UsesGpuRayGeneration(), "PreferGpuRayGeneration = false must generate rays on the CPU.");

    config.PreferGpuRayGeneration = true;
    raylidar::Simulator gpu(config);
    const bool shaderFound = std::filesystem::is_regular_file(fixture.TraceShader.parent_path() / "LidarRayGen.spv");
    Check(gpu.UsesGpuRayGeneration() == shaderFound, "GPU ray generation must be used exactly when its shader exists.");

    const auto cpuOutput = cpu.Scan(kIdentity);
    const auto gpuOutput = gpu.Scan(kIdentity);
    CheckWallScan(cpu, cpuOutput);
    CheckWallScan(gpu, gpuOutput);

    std::size_t different = 0;
    for (std::size_t index = 0; index < cpuOutput.Ranges.size(); ++index)
    {
        different += std::abs(cpuOutput.Ranges[index] - gpuOutput.Ranges[index]) > 1.0e-3f ? 1 : 0;
    }
    Check(different <= kEdgeTolerance, std::to_string(different) + " rays differ between CPU and GPU generation.");
}

void GaussianModeScansWall(const Fixture &fixture)
{
    raylidar::Simulator simulator(fixture.Config(raylidar::SceneMode::Gaussian));
    const auto output = simulator.Scan(kIdentity);
    CheckWallScan(simulator, output);
    Check(!output.Points.empty(), "Gaussian mode found no hits.");
    Check(Count(output.Sources, raylidar::HitSource::Gaussian) == output.Points.size(),
          "Gaussian hits must come from the Gaussians.");
}

void HybridModeTrustsAgreeingWall(const Fixture &fixture)
{
    raylidar::Simulator simulator(fixture.Config(raylidar::SceneMode::Hybrid));
    const auto output = simulator.Scan(kIdentity);
    CheckWallScan(simulator, output);

    // The mesh answers every ray it hits; the splats agree with it there, so those hits are trusted.
    const auto meshHits = Count(output.Sources, raylidar::HitSource::Mesh);
    Check(Near(meshHits, kWallHits, kEdgeTolerance), "Hybrid lost mesh hits: " + std::to_string(meshHits));
    Check(Count(output.Trust, raylidar::SurfaceTrust::Trusted) == meshHits,
          "Every mesh hit on the agreeing wall must be trusted.");
}

void MeshModeWithGaussiansAddsTrustLabels(const Fixture &fixture)
{
    raylidar::Simulator simulator(fixture.Config(raylidar::SceneMode::Mesh));
    const auto output = simulator.Scan(kIdentity);
    CheckWallScan(simulator, output);
    Check(Count(output.Sources, raylidar::HitSource::Gaussian) == 0, "Mesh mode must not trace the Gaussians.");
    Check(Count(output.Trust, raylidar::SurfaceTrust::Trusted) == output.Points.size(),
          "Mesh mode with Gaussians must label every hit on the agreeing wall as trusted.");
}

void WriteFrameMatchesFormat(const Fixture &fixture)
{
    raylidar::Simulator simulator(fixture.MeshPath, fixture.TraceShader);
    auto output = simulator.Scan(kIdentity);

    // .rlf: 128-byte header, then range f32, point f32x3, source u8 and trust u8 per ray.
    const auto path = fixture.Directory / "frame.rlf";
    raylidar::Simulator::WriteFrame(path, output, 7);
    Check(std::filesystem::file_size(path) == 128 + output.Ranges.size() * (4 + 12 + 1 + 1), "Unexpected .rlf size.");

    output.Trust.pop_back();
    CheckThrows<std::invalid_argument>([&] { raylidar::Simulator::WriteFrame(path, output); },
                                       "WriteFrame with a short Trust array");
}

void InvalidConfigsAreRejected(const Fixture &fixture)
{
    using raylidar::SceneMode;
    const auto rejects = [](const raylidar::SimulatorConfig &config, const std::string &what) {
        CheckThrows<std::invalid_argument>([&] { raylidar::Simulator simulator(config); }, what);
    };

    auto config = fixture.Config(SceneMode::Mesh);
    config.MeshPath.clear();
    rejects(config, "Mesh mode without a mesh");

    config = fixture.Config(SceneMode::Mesh);
    config.MeshPath = fixture.Directory / "missing.ply";
    rejects(config, "Missing mesh file");

    config = fixture.Config(SceneMode::Hybrid);
    config.GaussianPath.clear();
    rejects(config, "Hybrid mode without Gaussians");

    config = fixture.Config(SceneMode::Gaussian);
    config.ShaderDirectory = fixture.Directory;
    rejects(config, "Missing Gaussian shader");

    config = fixture.Config(SceneMode::Gaussian);
    config.Gaussian.AlphaThreshold = 1.0f;
    rejects(config, "AlphaThreshold of 1");

    config = fixture.Config(SceneMode::Gaussian);
    config.Gaussian.SigmaCutoff = std::numeric_limits<float>::quiet_NaN();
    rejects(config, "NaN SigmaCutoff");

    config = fixture.Config(SceneMode::Mesh);
    config.Sensor.BeamCount = 0;
    rejects(config, "Zero beams");

    config = fixture.Config(SceneMode::Mesh);
    config.Sensor.BeamCount = 1u << 16;
    config.Sensor.AzimuthCount = 1u << 16;
    rejects(config, "Too many rays");

    config = fixture.Config(SceneMode::Mesh);
    config.Sensor.ElevationMinDegrees = 10.0f;
    rejects(config, "ElevationMinDegrees above ElevationMaxDegrees");

    config = fixture.Config(SceneMode::Mesh);
    config.Sensor.RangeMax = config.Sensor.RangeMin;
    rejects(config, "Empty range interval");

    config = fixture.Config(static_cast<SceneMode>(7));
    rejects(config, "Unknown SceneMode");

    // Every problem is reported at once.
    config = fixture.Config(SceneMode::Hybrid);
    config.GaussianPath.clear();
    config.Sensor.BeamCount = 0;
    try
    {
        raylidar::Simulator simulator(config);
        throw std::runtime_error("Two config problems: did not throw.");
    }
    catch (const std::invalid_argument &error)
    {
        const std::string message = error.what();
        Check(message.find("GaussianPath") != std::string::npos && message.find("BeamCount") != std::string::npos,
              "The error must list every problem, got: " + message);
    }
}

void InvalidPosesAreRejected(const Fixture &fixture)
{
    raylidar::Simulator simulator(fixture.MeshPath, fixture.TraceShader);
    const auto rejects = [&simulator](const raylidar::Pose &pose, const std::string &what) {
        CheckThrows<std::invalid_argument>([&] { static_cast<void>(simulator.Scan(pose)); }, what);
    };

    auto pose = kIdentity;
    pose[0][0] = 2.0;
    rejects(pose, "Scaled pose");

    pose = kIdentity;
    pose[1][1] = -1.0;
    rejects(pose, "Reflected pose");

    pose = kIdentity;
    pose[3][0] = 1.0;
    rejects(pose, "Projective last row");

    pose = kIdentity;
    pose[0][3] = std::numeric_limits<double>::infinity();
    rejects(pose, "Infinite translation");

    // A valid pose still works after the rejected ones.
    pose = kIdentity;
    pose[2][3] = 1.5;
    CheckWallScan(simulator, simulator.Scan(pose));
}

void MovedFromSimulatorIsDetected(const Fixture &fixture)
{
    raylidar::Simulator source(fixture.MeshPath, fixture.TraceShader);
    raylidar::Simulator target = std::move(source);
    CheckWallScan(target, target.Scan(kIdentity));
    CheckThrows<std::logic_error>([&] { static_cast<void>(source.Scan(kIdentity)); }, // NOLINT(bugprone-use-after-move)
                                  "Scan on a moved-from Simulator");
}

struct TestCase
{
    const char *Name;
    void (*Run)(const Fixture &);
};

constexpr std::array kTests = {
    TestCase{"Mesh mode scans the wall", MeshModeScansWall},
    TestCase{"CPU and GPU ray generation agree", CpuAndGpuRayGenerationAgree},
    TestCase{"Gaussian mode scans the wall", GaussianModeScansWall},
    TestCase{"Hybrid mode trusts the agreeing wall", HybridModeTrustsAgreeingWall},
    TestCase{"Mesh mode with Gaussians adds trust labels", MeshModeWithGaussiansAddsTrustLabels},
    TestCase{"WriteFrame matches the .rlf format", WriteFrameMatchesFormat},
    TestCase{"Invalid configs are rejected", InvalidConfigsAreRejected},
    TestCase{"Invalid poses are rejected", InvalidPosesAreRejected},
    TestCase{"Moved-from Simulator is detected", MovedFromSimulatorIsDetected},
};
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
        const TempDirectory directory;
        Fixture fixture;
        fixture.TraceShader = argv[1];
        fixture.Directory = directory.Path();
        fixture.MeshPath = directory.Path() / "wall.ply";
        fixture.SplatPath = directory.Path() / "splats.ply";
        WriteWallPly(fixture.MeshPath);
        WriteSplatWallPly(fixture.SplatPath);

        int failures = 0;
        for (const auto &test : kTests)
        {
            try
            {
                test.Run(fixture);
                std::cout << "[ PASS ] " << test.Name << '\n';
            }
            catch (const std::exception &exception)
            {
                std::cout << "[ FAIL ] " << test.Name << ": " << exception.what() << '\n';
                ++failures;
            }
        }

        std::cout << kTests.size() - static_cast<std::size_t>(failures) << " / " << kTests.size()
                  << " public API tests passed.\n";
        return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    catch (const std::exception &exception)
    {
        std::cerr << "Public API test setup failed: " << exception.what() << '\n';
        return EXIT_FAILURE;
    }
}
