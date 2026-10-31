#include "fixture.h"

#include <raylidar/cli/commands_phase2.h>
#include <raylidar/common/io.h>
#include <raylidar/gt/cache.h>

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

namespace
{
namespace fs = std::filesystem;
using raylidar::IoError;
using raylidar::ReadJson;
using raylidar::Sha256File;
using raylidar::gt::GtPoint;
using raylidar::gt::ReadGtFrame;
using raylidar::test::MakeSyntheticGtTree;
using raylidar::test::SyntheticGtTree;
using raylidar::test::WriteGtRawBytes;
using raylidar::test::WriteGtTextFile;
using raylidar::test::WriteSyntheticSegmentManifest;

int RunGtCacheWith(const std::vector<std::string> &arguments)
{
    std::vector<std::string_view> views;
    views.reserve(arguments.size());
    for (const std::string &argument : arguments)
    {
        views.emplace_back(argument);
    }
    return RunGtCache(std::span<const std::string_view>(views.data(), views.size()));
}

std::vector<std::uint8_t> ReadAllBytes(const fs::path &path)
{
    std::ifstream input(path, std::ios::binary);
    return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(input),
                                     std::istreambuf_iterator<char>());
}

fs::path FramePath(const SyntheticGtTree &tree, const int frame)
{
    return tree.phaseDir / "gt" /
           ("frame_" + std::string(10U - std::to_string(frame).size(), '0') +
            std::to_string(frame) + ".bin");
}
} // namespace

TEST_CASE("gt-cache smoke run writes the expected kept-static window", "[gt]")
{
    const SyntheticGtTree tree = MakeSyntheticGtTree("gt_cli_ok");
    CHECK(RunGtCacheWith({"--config", tree.configPath.string(), "--frames", "0,1"}) == 0);

    const fs::path manifestPath = tree.phaseDir / "gt" / "gt_manifest.smoke.json";
    REQUIRE(fs::exists(manifestPath));
    CHECK_FALSE(fs::exists(tree.phaseDir / "gt" / "gt_manifest.json"));

    const nlohmann::json manifest = ReadJson(manifestPath);
    REQUIRE(manifest["frames"].size() == tree.frames.size());
    CHECK(manifest["totals"]["points"].get<std::size_t>() ==
          tree.frames.size() * tree.keptPerFrame);

    for (std::size_t index = 0; index < tree.frames.size(); ++index)
    {
        const int frame = tree.frames[index];
        const nlohmann::json &entry = manifest["frames"][index];
        CHECK(entry["frame"].get<int>() == frame);
        CHECK(entry["count"].get<std::size_t>() == tree.keptPerFrame);
        CHECK(entry["camera_00"].get<std::size_t>() == tree.keptPerFrame);
        CHECK(entry["camera_01"].get<std::size_t>() == 0U);
        CHECK(entry["camera_unknown"].get<std::size_t>() == 0U);

        const fs::path path = FramePath(tree, frame);
        REQUIRE(fs::exists(path));
        CHECK(fs::file_size(path) == 64U + tree.keptPerFrame * 32U);
        CHECK(Sha256File(path) == entry["sha256"].get<std::string>());

        const std::vector<std::uint8_t> bytes = ReadAllBytes(path);
        REQUIRE(bytes.size() >= 4U);
        CHECK(std::string(reinterpret_cast<const char *>(bytes.data()), 4U) == "RLGT");

        int decodedFrame = -1;
        const std::vector<GtPoint> points = ReadGtFrame(path, &decodedFrame);
        CHECK(decodedFrame == frame);
        REQUIRE(points.size() == tree.keptPerFrame);
        for (const GtPoint &point : points)
        {
            CHECK(point.camera == 0U);
            CHECK(point.valid == 1U);
            CHECK(point.semanticId == 7U);
            CHECK(point.r_gt > 0.0F);
        }
    }
}

TEST_CASE("gt-cache rejects a non-full segment manifest", "[gt]")
{
    const SyntheticGtTree tree = MakeSyntheticGtTree("gt_cli_smoke_manifest");
    WriteSyntheticSegmentManifest(tree.segmentManifestPath, tree.frames,
                                  static_cast<int>(tree.frames.size()), "smoke");
    CHECK(RunGtCacheWith({"--config", tree.configPath.string()}) == 4);
    CHECK_FALSE(fs::exists(tree.phaseDir / "gt" / "gt_manifest.json"));
}

TEST_CASE("gt-cache rejects a selected.frames length mismatch", "[gt]")
{
    const SyntheticGtTree tree = MakeSyntheticGtTree("gt_cli_length");
    WriteSyntheticSegmentManifest(tree.segmentManifestPath, tree.frames, 3, "full");
    CHECK(RunGtCacheWith({"--config", tree.configPath.string()}) == 4);
}

TEST_CASE("gt-cache rejects a non-ascending selected.frames list", "[gt]")
{
    const SyntheticGtTree tree = MakeSyntheticGtTree("gt_cli_order");
    WriteSyntheticSegmentManifest(tree.segmentManifestPath, {1, 0}, 2, "full");
    CHECK(RunGtCacheWith({"--config", tree.configPath.string()}) == 4);
}

TEST_CASE("gt-cache exits 4 without the successor pose", "[gt]")
{
    const SyntheticGtTree tree = MakeSyntheticGtTree("gt_cli_pose");
    WriteGtTextFile(tree.posesPath, "0 1 0 0 0 0 1 0 0 0 0 1 0\n");
    CHECK(RunGtCacheWith({"--config", tree.configPath.string(), "--frames", "0"}) == 4);
}

TEST_CASE("gt-cache exits 4 without the successor timestamp", "[gt]")
{
    const SyntheticGtTree tree = MakeSyntheticGtTree("gt_cli_time");
    WriteGtTextFile(tree.timestampsPath, "2013-05-28 00:00:00.000000000\n");
    CHECK(RunGtCacheWith({"--config", tree.configPath.string(), "--frames", "0"}) == 4);
}

TEST_CASE("gt-cache detects a tampered GT frame", "[gt]")
{
    const SyntheticGtTree tree = MakeSyntheticGtTree("gt_cli_tamper");
    REQUIRE(RunGtCacheWith({"--config", tree.configPath.string(), "--frames", "0,1"}) == 0);

    const nlohmann::json manifest =
        ReadJson(tree.phaseDir / "gt" / "gt_manifest.smoke.json");
    const std::string recorded = manifest["frames"][0]["sha256"].get<std::string>();
    const fs::path path = FramePath(tree, tree.frames.front());
    CHECK(Sha256File(path) == recorded);

    std::vector<std::uint8_t> bytes = ReadAllBytes(path);
    REQUIRE(bytes.size() > 70U);
    bytes[70] ^= 0xFFU;
    WriteGtRawBytes(path, bytes);
    CHECK(Sha256File(path) != recorded);

    bytes[0] = 'X';
    bytes[1] = 'X';
    bytes[2] = 'X';
    bytes[3] = 'X';
    WriteGtRawBytes(path, bytes);
    CHECK_THROWS_AS(ReadGtFrame(path), IoError);
}
