#include "fixture.h"

#include <raylidar/cli/commands.h>
#include <raylidar/common/io.h>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace
{
namespace fs = std::filesystem;
using raylidar::ReadJson;
using raylidar::test::MakeOverlayTempRoot;
using raylidar::test::MakeOverlayTree;
using raylidar::test::OverlayTreeOptions;
using raylidar::test::WriteOverlayConfig;

constexpr const char *kSequence = "2013_05_28_drive_0000_sync";

struct Invocation
{
    int code = 0;
    std::string out;
    std::string err;
};

Invocation Invoke(const std::vector<std::string> &args)
{
    std::vector<std::string_view> views;
    views.reserve(args.size());
    for (const std::string &arg : args)
    {
        views.emplace_back(arg);
    }
    Invocation result;
    std::ostringstream out;
    std::ostringstream err;
    std::streambuf *oldOut = std::cout.rdbuf(out.rdbuf());
    std::streambuf *oldErr = std::cerr.rdbuf(err.rdbuf());
    try
    {
        result.code = RunOverlayCheck(views);
    }
    catch (...)
    {
        std::cout.rdbuf(oldOut);
        std::cerr.rdbuf(oldErr);
        throw;
    }
    std::cout.rdbuf(oldOut);
    std::cerr.rdbuf(oldErr);
    result.out = out.str();
    result.err = err.str();
    return result;
}

struct FixtureRun
{
    fs::path root;
    fs::path outDir;
    Invocation invocation;
};

FixtureRun RunFixture(const OverlayTreeOptions &options, const std::vector<std::string> &extra, std::string_view name)
{
    FixtureRun run;
    run.root = MakeOverlayTree(MakeOverlayTempRoot(name), options);
    const fs::path config = WriteOverlayConfig(run.root / "frames.yaml", run.root, kSequence);
    run.outDir = run.root / "out";
    std::vector<std::string> args = {"--config", config.string(), "--out-dir", run.outDir.string()};
    args.insert(args.end(), extra.begin(), extra.end());
    run.invocation = Invoke(args);
    return run;
}
} // namespace

TEST_CASE("overlay-check writes both cameras, PNGs, and a report", "[overlay]")
{
    const FixtureRun run = RunFixture({}, {"--frames", "0"}, "cli_ok");
    CHECK(run.invocation.code == 0);
    CHECK(run.invocation.out.find("ADVISORY ONLY") != std::string::npos);

    const fs::path frameDir = run.outDir / "frame_0000000000";
    for (const char *camera : {"image_00", "image_01"})
    {
        CHECK(fs::is_regular_file(frameDir / (std::string(camera) + "_overlay_1x.png")));
        CHECK(fs::is_regular_file(frameDir / (std::string(camera) + "_overlay_4x.png")));
        CHECK(fs::is_regular_file(frameDir / (std::string(camera) + "_overlay_report.json")));
    }

    const nlohmann::json aggregate = ReadJson(run.outDir / "overlay_report.json");
    CHECK(aggregate["schema_version"] == 1);
    CHECK(aggregate["sequence"] == kSequence);
    CHECK(aggregate["advisory_only"] == true);
    CHECK(aggregate["rectified_data_rect_only"] == true);
    CHECK(aggregate["results"].size() == 2U);
    REQUIRE(aggregate.contains("generated_at_utc"));

    for (const nlohmann::json &result : aggregate["results"])
    {
        CHECK(result["frame"] == 0);
        CHECK(result["rectified"] == true);
        CHECK(result["image"].get<std::string>().find("data_rect") != std::string::npos);
        CHECK(result["image_resolved"].get<std::string>().find("data_rect") != std::string::npos);
        CHECK(result["scale_factor"] == 4);
        CHECK(result.contains("median_px"));
        CHECK(result.contains("p90_px"));
        CHECK(result.contains("count"));
        CHECK(result.contains("fraction_within_2px"));
        CHECK(result["subsampled"] == false);
    }
}

TEST_CASE("overlay-check --help lists its options", "[overlay]")
{
    const Invocation invocation = Invoke({"--help"});
    CHECK(invocation.code == 0);
    for (const char *option : {"--frames", "--auto-frames", "--cams", "--out-dir", "--max-points"})
    {
        CHECK(invocation.out.find(option) != std::string::npos);
    }
}

TEST_CASE("overlay-check rejects unsupported cameras", "[overlay]")
{
    const FixtureRun run = RunFixture({}, {"--frames", "0", "--cams", "image_02"}, "cli_bad_cam");
    CHECK(run.invocation.code == 4);
    CHECK(run.invocation.err.find("unsupported camera") != std::string::npos);
}

TEST_CASE("overlay-check rejects frames that do not exist", "[overlay]")
{
    const FixtureRun run = RunFixture({}, {"--frames", "5"}, "cli_missing_frame");
    CHECK(run.invocation.code == 4);
    CHECK(run.invocation.err.find("frame 5") != std::string::npos);
}

TEST_CASE("overlay-check rejects non-data_rect image provenance", "[overlay]")
{
    OverlayTreeOptions options;
    options.linkRectToRaw = true;
    const FixtureRun run = RunFixture(options, {"--frames", "0"}, "cli_raw_image");
    CHECK(run.invocation.code == 4);
    CHECK(run.invocation.err.find("rectified_data_rect_only") != std::string::npos);
}

TEST_CASE("overlay-check rejects a malformed LiDAR scan", "[overlay]")
{
    OverlayTreeOptions options;
    options.malformedLidar = true;
    const FixtureRun run = RunFixture(options, {"--frames", "0"}, "cli_bad_lidar");
    CHECK(run.invocation.code == 4);
    CHECK(run.invocation.err.find("LiDAR scan has an invalid size") != std::string::npos);
}

TEST_CASE("overlay-check subsamples large scans deterministically", "[overlay]")
{
    OverlayTreeOptions options;
    options.pointCount = 200;
    const FixtureRun run = RunFixture(options, {"--frames", "0", "--max-points", "100"}, "cli_subsample");
    CHECK(run.invocation.code == 0);

    const nlohmann::json aggregate = ReadJson(run.outDir / "overlay_report.json");
    REQUIRE(aggregate["results"].size() == 2U);
    for (const nlohmann::json &result : aggregate["results"])
    {
        CHECK(result["subsampled"] == true);
        CHECK(result["points_total"] == 100);
    }
}
