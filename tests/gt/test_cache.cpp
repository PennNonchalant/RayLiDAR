#include "fixture.h"

#include <raylidar/common/io.h>
#include <raylidar/gt/cache.h>
#include <raylidar/gt/manifest.h>

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace
{
namespace fs = std::filesystem;
using raylidar::IoError;
using raylidar::ReadJson;
using raylidar::gt::BuildGtManifest;
using raylidar::gt::GtPoint;
using raylidar::gt::ReadGtFrame;
using raylidar::gt::WriteGtFrame;
using raylidar::test::MakeGtTempRoot;
using raylidar::test::WriteGtRawBytes;

std::vector<std::uint8_t> ReadAllBytes(const fs::path &path)
{
    std::ifstream input(path, std::ios::binary);
    return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

GtPoint MakePoint(const float x, const float y, const float z, const std::uint32_t frame, const std::uint16_t label,
                  const std::uint8_t camera, const float range)
{
    GtPoint point;
    point.p_L = Eigen::Vector3f(x, y, z);
    point.frame = frame;
    point.semanticId = label;
    point.camera = camera;
    point.valid = 1U;
    point.r_gt = range;
    return point;
}
} // namespace

TEST_CASE("the GT header and records round-trip exactly", "[gt]")
{
    const fs::path root = MakeGtTempRoot("cache_roundtrip");
    const fs::path path = root / "frame_0000000007.bin";
    const std::vector<GtPoint> points = {MakePoint(1.0F, 2.0F, 3.0F, 7U, 26U, 0U, 4.5F),
                                         MakePoint(-1.0F, 0.5F, 10.0F, 7U, 65535U, 255U, -1.0F)};

    WriteGtFrame(path, 7, points);
    const std::vector<std::uint8_t> bytes = ReadAllBytes(path);
    REQUIRE(bytes.size() == 64U + points.size() * 32U);
    CHECK(std::string(reinterpret_cast<const char *>(bytes.data()), 4U) == "RLGT");

    int frame = -1;
    const std::vector<GtPoint> decoded = ReadGtFrame(path, &frame);
    CHECK(frame == 7);
    REQUIRE(decoded.size() == points.size());
    for (std::size_t index = 0; index < points.size(); ++index)
    {
        CHECK(decoded[index].p_L.isApprox(points[index].p_L));
        CHECK(decoded[index].frame == points[index].frame);
        CHECK(decoded[index].semanticId == points[index].semanticId);
        CHECK(decoded[index].camera == points[index].camera);
        CHECK(decoded[index].valid == points[index].valid);
        CHECK(decoded[index].r_gt == points[index].r_gt);
    }
}

TEST_CASE("ReadGtFrame rejects a truncated file", "[gt]")
{
    const fs::path root = MakeGtTempRoot("cache_truncated");
    const fs::path path = root / "frame_0000000001.bin";
    WriteGtFrame(path, 1, {MakePoint(0.0F, 0.0F, 10.0F, 1U, 7U, 0U, 10.0F)});

    std::vector<std::uint8_t> bytes = ReadAllBytes(path);
    REQUIRE(!bytes.empty());
    bytes.pop_back();
    WriteGtRawBytes(path, bytes);

    bool threw = false;
    try
    {
        (void)ReadGtFrame(path);
    }
    catch (const IoError &error)
    {
        threw = true;
        CHECK(error.exitCode() == 4);
    }
    CHECK(threw);
}

TEST_CASE("ReadGtFrame rejects a wrong magic", "[gt]")
{
    const fs::path root = MakeGtTempRoot("cache_magic");
    const fs::path path = root / "frame_0000000002.bin";
    WriteGtFrame(path, 2, {MakePoint(0.0F, 0.0F, 10.0F, 2U, 7U, 0U, 10.0F)});

    std::vector<std::uint8_t> bytes = ReadAllBytes(path);
    REQUIRE(bytes.size() > 4U);
    bytes[0] = 'X';
    bytes[1] = 'X';
    bytes[2] = 'X';
    bytes[3] = 'X';
    WriteGtRawBytes(path, bytes);

    CHECK_THROWS_AS(ReadGtFrame(path), IoError);
}

TEST_CASE("BuildGtManifest records per-frame hashes and totals", "[gt]")
{
    const fs::path root = MakeGtTempRoot("cache_manifest");
    const fs::path phaseDir = root / "artifacts" / "02";

    nlohmann::json entries = nlohmann::json::array();
    for (const int frame : {0, 1})
    {
        const fs::path path = phaseDir / "gt" / ("frame_" + std::to_string(frame) + ".bin");
        WriteGtFrame(path, frame, {MakePoint(0.0F, 0.0F, 10.0F, static_cast<std::uint32_t>(frame), 7U, 0U, 10.0F)});
        nlohmann::json entry;
        entry["frame"] = frame;
        entry["count"] = 1;
        entry["camera_00"] = 1;
        entry["camera_01"] = 0;
        entry["camera_unknown"] = 0;
        entry["sha256"] = raylidar::Sha256File(path);
        entry["config_sha256"] = "abc123";
        entries.push_back(entry);
    }

    BuildGtManifest(phaseDir, "seq", entries);
    const nlohmann::json manifest = ReadJson(phaseDir / "gt" / "gt_manifest.json");
    CHECK(manifest["schema_version"].get<int>() == 1);
    CHECK(manifest["sequence"].get<std::string>() == "seq");
    CHECK(manifest["config_sha256"].get<std::string>() == "abc123");
    REQUIRE(manifest["frames"].size() == 2U);
    CHECK_FALSE(manifest["frames"][0].contains("config_sha256"));
    CHECK(manifest["totals"]["frames"].get<std::size_t>() == 2U);
    CHECK(manifest["totals"]["points"].get<std::size_t>() == 2U);
    CHECK(manifest["totals"]["camera_00"].get<std::size_t>() == 2U);
}
