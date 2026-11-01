#include "fixture.h"

#include <raylidar/common/io.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>

namespace raylidar::test
{
namespace fs = std::filesystem;

Phase3Fixture::Phase3Fixture(const int frames, const int gridWidth, const int gridHeight)
{
    static std::uint64_t counter = 0;
    static std::random_device device;
    root = fs::temp_directory_path() /
           ("raylidar-eval-cli-" + std::to_string(device()) + "-" +
            std::to_string(counter++));

    const fs::path configDir = root / "cfg";
    fs::create_directories(configDir);
    outDir = root / "out";
    configPath = configDir / "phase3.yaml";
    protocolPath = configDir / "protocol.md";

    WriteFileText(protocolPath,
                  "# Test protocol document\n"
                  "\n"
                  "Tiny protocol used by the eval CLI fixture. The real frozen\n"
                  "contract lives in docs/protocol.md.\n");

    std::ostringstream config;
    config << "schema_version: 1\n"
           << "phase: 3\n"
           << "run_id: p3-fixture\n"
           << "artifacts:\n"
           << "  phase_dir: " << (root / "artifacts").string() << "\n"
           << "protocol:\n"
           << "  protocol_version: 1\n"
           << "  t_min_m: 1.0e-4\n"
           << "  r_min_m: 1.5\n"
           << "  t_max_m: 60.0\n"
           << "  taus_m: [0.05, 0.20, 0.50]\n"
           << "  primary_tau_m: 0.20\n"
           << "  primary_buckets: [[0.0, 20.0], [20.0, 40.0], [40.0, 50.0]]\n"
           << "  secondary_bucket: [50.0, 60.0]\n"
           << "placeholder:\n"
           << "  generator: procedural_street_v1\n"
           << "  seed: 20261101\n"
           << "  ground_length_m: 40.0\n"
           << "  ground_width_m: 16.0\n"
           << "  box_count: 6\n"
           << "  thin_box_count: 3\n"
           << "  cameras:\n"
           << "    - id: 0\n"
           << "      origin_m: [0.0, 0.0, 1.6]\n"
           << "    - id: 1\n"
           << "      origin_m: [0.54, 0.0, 1.6]\n"
           << "  ray_grid:\n"
           << "    width: " << gridWidth << "\n"
           << "    height: " << gridHeight << "\n"
           << "    hfov_deg: 90.0\n"
           << "    vfov_deg: 30.0\n"
           << "  frames: " << frames << "\n"
           << "  frame_stride_m: 0.5\n"
           << "  predictor:\n"
           << "    name: linear_range_bias_scale_v1\n"
           << "    range_bias_m: 0.02\n"
           << "    range_scale: 0.01\n"
           << "    miss_every_n: 7\n";
    WriteFileText(configPath, config.str());
}

Phase3Fixture::~Phase3Fixture()
{
    std::error_code error;
    fs::remove_all(root, error);
}

std::string ReadFileText(const fs::path &path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input)
    {
        throw std::runtime_error("cannot read " + path.string());
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

void WriteFileText(const fs::path &path, const std::string &text)
{
    raylidar::EnsureDir(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output)
    {
        throw std::runtime_error("cannot write " + path.string());
    }
    output.write(text.data(), static_cast<std::streamsize>(text.size()));
    if (!output)
    {
        throw std::runtime_error("failed to write " + path.string());
    }
}
} // namespace raylidar::test
