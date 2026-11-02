#include <raylidar/common/artifacts.h>
#include <raylidar/common/io.h>

#include <stb_image.h>
#include <stb_image_write.h>

#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using raylidar::ArtifactsRoot;
using raylidar::CanonicalJson;
using raylidar::EnsureDir;
using raylidar::IoError;
using raylidar::PhaseArtifactsDir;
using raylidar::ReadJson;
using raylidar::ReadYaml;
using raylidar::Sha256String;
using raylidar::WriteJson;

namespace
{
std::filesystem::path TestDir()
{
    const auto dir = std::filesystem::temp_directory_path() / "raylidar_test_io";
    EnsureDir(dir);
    return dir;
}
} // namespace

TEST_CASE("Sha256String matches known vectors", "[common]")
{
    CHECK(Sha256String("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    CHECK(Sha256String("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
}

TEST_CASE("WriteJson and ReadJson round-trip", "[common]")
{
    const auto path = TestDir() / "roundtrip.json";
    nlohmann::json value;
    value["name"] = "raylidar";
    value["numbers"] = {1, 2, 3};
    value["nested"]["enabled"] = true;

    WriteJson(path, value);
    const auto loaded = ReadJson(path);
    CHECK(loaded == value);

    std::ifstream in(path, std::ios::binary);
    const std::string contents((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    CHECK(contents.back() == '\n');
    CHECK(contents.find("\n  \"name\"") != std::string::npos);
}

TEST_CASE("CanonicalJson sorts keys deterministically", "[common]")
{
    nlohmann::json first;
    first["zebra"] = 26;
    first["alpha"] = 1;
    first["middle"]["b"] = 2;
    first["middle"]["a"] = 1;

    nlohmann::json second;
    second["middle"]["a"] = 1;
    second["middle"]["b"] = 2;
    second["alpha"] = 1;
    second["zebra"] = 26;

    CHECK(CanonicalJson(first).dump() == CanonicalJson(second).dump());
    CHECK(CanonicalJson(first).dump() == "{\"alpha\":1,\"middle\":{\"a\":1,\"b\":2},\"zebra\":26}");
}

TEST_CASE("ReadYaml handles empty and malformed input", "[common]")
{
    const auto emptyPath = TestDir() / "empty.yaml";
    {
        std::ofstream out(emptyPath, std::ios::binary | std::ios::trunc);
        out << "";
    }
    const auto empty = ReadYaml(emptyPath);
    CHECK(empty.IsMap());
    CHECK(empty.size() == 0U);

    const auto malformedPath = TestDir() / "malformed.yaml";
    {
        std::ofstream out(malformedPath, std::ios::binary | std::ios::trunc);
        out << "a: [1, 2\n";
    }
    try
    {
        (void)ReadYaml(malformedPath);
        FAIL("malformed YAML did not throw");
    }
    catch (const IoError &error)
    {
        CHECK(error.exitCode() == 4);
    }
}

TEST_CASE("STB implementation is linked exactly once", "[common]")
{
    unsigned char pixels[2 * 2 * 3] = {
        255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255,
    };
    std::vector<unsigned char> png;
    const int wrote = stbi_write_png_to_func(
        [](void *context, void *data, int size) {
            auto *out = static_cast<std::vector<unsigned char> *>(context);
            const auto *bytes = static_cast<const unsigned char *>(data);
            out->insert(out->end(), bytes, bytes + size);
        },
        &png, 2, 2, 3, pixels, 2 * 3);
    REQUIRE(wrote != 0);
    REQUIRE_FALSE(png.empty());

    int width = 0;
    int height = 0;
    int channels = 0;
    CHECK(stbi_info_from_memory(png.data(), static_cast<int>(png.size()), &width, &height, &channels) == 1);
    CHECK(width == 2);
    CHECK(height == 2);
}

TEST_CASE("ArtifactsRoot honours the environment and phase padding", "[common]")
{
#if defined(_WIN32)
    _putenv_s("RAYLIDAR_ARTIFACTS_ROOT", "/tmp/raylidar-artifacts-test");
#else
    ::setenv("RAYLIDAR_ARTIFACTS_ROOT", "/tmp/raylidar-artifacts-test", 1);
#endif
    CHECK(ArtifactsRoot() == std::filesystem::path("/tmp/raylidar-artifacts-test"));
    CHECK(PhaseArtifactsDir(1) == std::filesystem::path("/tmp/raylidar-artifacts-test") / "01");

#if defined(_WIN32)
    _putenv_s("RAYLIDAR_ARTIFACTS_ROOT", "");
#else
    ::unsetenv("RAYLIDAR_ARTIFACTS_ROOT");
#endif
    CHECK(ArtifactsRoot() == std::filesystem::path("/data-20T/hybrid-lidar-sim/artifacts"));
}
