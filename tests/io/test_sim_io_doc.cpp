#include <raylidar/common/io.h>
#include <raylidar/io/sim_io.h>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>

#ifndef RAYLIDAR_SOURCE_DIR
#define RAYLIDAR_SOURCE_DIR "."
#endif

namespace
{
namespace fs = std::filesystem;

std::string ReadText(const fs::path &path)
{
    std::ifstream input(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}
} // namespace

TEST_CASE("docs/sim-io.md carries the frozen contract tokens", "[io][doc]")
{
    const fs::path docPath = fs::path(RAYLIDAR_SOURCE_DIR) / "docs" / "sim-io.md";
    const std::string text = ReadText(docPath);
    REQUIRE_FALSE(text.empty());

    const char *tokens[] = {"RLGR",
                            "RLHT",
                            "origin_W",
                            "dir_W",
                            "t_gt",
                            "t_min",
                            "t_max",
                            "ray_id",
                            "t_pred",
                            "prim_id",
                            "little-endian",
                            "binary_little_endian",
                            "schema_version",
                            "sim_io_manifest.json",
                            "PLY",
                            "GLB",
                            "OBJ",
                            "metres",
                            "W frame"};
    for (const char *token : tokens)
    {
        INFO(token);
        CHECK(text.find(token) != std::string::npos);
    }
    CHECK(text.find("Out of scope") != std::string::npos);
    CHECK(text.find("IO-02") != std::string::npos);
    CHECK(text.find("IO-03") != std::string::npos);
    CHECK(text.find("GPU Ray Query") != std::string::npos);
    CHECK(text.find("plots") != std::string::npos);
    CHECK(text.find("reports") != std::string::npos);
}

TEST_CASE("docs/sim-io.sha256 pins the document", "[io][doc]")
{
    const fs::path docPath = fs::path(RAYLIDAR_SOURCE_DIR) / "docs" / "sim-io.md";
    const fs::path sidecarPath = fs::path(RAYLIDAR_SOURCE_DIR) / "docs" / "sim-io.sha256";

    const std::string text = ReadText(sidecarPath);
    std::istringstream stream(text);
    std::string digest;
    std::string filename;
    stream >> digest >> filename;

    REQUIRE_FALSE(digest.empty());
    CHECK(digest == raylidar::Sha256File(docPath));
    CHECK(filename == "docs/sim-io.md");
}

TEST_CASE("code constants agree with the frozen schema", "[io][doc]")
{
    using namespace raylidar::io;
    CHECK(kSimIoHeaderSize == 32U);
    CHECK(kGtRaysVersion == 1U);
    CHECK(kGtRaysRecordSize == 96U);
    CHECK(kHitsVersion == 1U);
    CHECK(kHitsRecordSize == 24U);
    CHECK(kPrimIdUnavailable == 0xFFFFFFFFU);
    CHECK(kMaxRecordCount == 200000000U);
}
