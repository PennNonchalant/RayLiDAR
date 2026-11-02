#pragma once

#include <filesystem>
#include <string>

namespace raylidar::test
{
/// Temporary source tree for one `raylidar_cli eval` end-to-end test.
///
/// Layout:
///   <root>/cfg/phase3.yaml    tiny deterministic placeholder config
///   <root>/cfg/protocol.md    protocol document resolved next to the config
///   <root>/out/               explicit --out-dir
struct Phase3Fixture
{
    explicit Phase3Fixture(int frames = 2, int gridWidth = 12, int gridHeight = 6);
    ~Phase3Fixture();

    Phase3Fixture(const Phase3Fixture &) = delete;
    Phase3Fixture &operator=(const Phase3Fixture &) = delete;

    std::filesystem::path root;
    std::filesystem::path configPath;
    std::filesystem::path protocolPath;
    std::filesystem::path outDir;
};

std::string ReadFileText(const std::filesystem::path &path);

void WriteFileText(const std::filesystem::path &path, const std::string &text);
} // namespace raylidar::test
