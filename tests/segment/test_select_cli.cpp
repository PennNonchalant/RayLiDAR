#include "fixture.h"

#include <raylidar/cli/commands_phase2.h>
#include <raylidar/common/io.h>

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace
{
namespace fs = std::filesystem;
using raylidar::ReadJson;
using raylidar::test::MakeSyntheticSegmentTree;
using raylidar::test::SyntheticSegmentTree;

struct Invocation
{
    int code = 0;
    std::string out;
    std::string err;
};

Invocation Invoke(const std::vector<std::string> &arguments)
{
    std::vector<std::string_view> views;
    views.reserve(arguments.size());
    for (const std::string &argument : arguments)
    {
        views.emplace_back(argument);
    }

    Invocation result;
    std::ostringstream out;
    std::ostringstream err;
    std::streambuf *oldOut = std::cout.rdbuf(out.rdbuf());
    std::streambuf *oldErr = std::cerr.rdbuf(err.rdbuf());
    try
    {
        result.code = RunSegmentSelect(std::span<const std::string_view>(views.data(), views.size()));
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

std::string FirstLine(const fs::path &path)
{
    std::ifstream input(path);
    std::string line;
    std::getline(input, line);
    return line;
}
} // namespace

TEST_CASE("segment-select writes a full manifest and candidate table", "[segment]")
{
    const SyntheticSegmentTree tree = MakeSyntheticSegmentTree("cli_full");
    const Invocation invocation = Invoke({"--config", tree.configPath.string()});
    REQUIRE(invocation.code == 0);
    CHECK(invocation.out.find("selected_start=") != std::string::npos);

    const fs::path manifestPath = tree.phaseDir / "segment" / "segment_manifest.json";
    const fs::path csvPath = tree.phaseDir / "segment" / "candidate_windows.csv";
    REQUIRE(fs::exists(manifestPath));
    REQUIRE(fs::exists(csvPath));
    CHECK_FALSE(fs::exists(tree.phaseDir / "segment" / "segment_manifest.smoke.json"));

    const nlohmann::json manifest = ReadJson(manifestPath);
    CHECK(manifest["selection_mode"] == "full");
    REQUIRE(manifest["selected"].is_object());
    const nlohmann::json &selected = manifest["selected"];
    CHECK(selected["frame_count"].get<std::size_t>() == selected["frames"].size());
    CHECK(selected["frames"].size() == 2U);
    CHECK(selected["frames"][0].get<int>() < selected["frames"][1].get<int>());
    CHECK_FALSE(selected["rationale"].get<std::string>().empty());
    CHECK(manifest["candidate_windows"].size() >= 2U);
    bool differs = false;
    for (const nlohmann::json &window : manifest["candidate_windows"])
    {
        if (window["start_frame"].get<int>() != selected["start_frame"].get<int>())
        {
            differs = true;
        }
    }
    CHECK(differs);

    CHECK(FirstLine(csvPath) == "start_frame,end_frame,frame_count,dynamic_points,static_points,score");
}

TEST_CASE("segment-select smoke mode never writes the final manifest", "[segment]")
{
    const SyntheticSegmentTree tree = MakeSyntheticSegmentTree("cli_smoke");
    const Invocation invocation = Invoke({"--config", tree.configPath.string(), "--max-frames", "3"});
    REQUIRE(invocation.code == 0);
    CHECK(fs::exists(tree.phaseDir / "segment" / "segment_manifest.smoke.json"));
    CHECK_FALSE(fs::exists(tree.phaseDir / "segment" / "segment_manifest.json"));
    CHECK_FALSE(fs::exists(tree.phaseDir / "segment" / "candidate_windows.csv"));
}

TEST_CASE("segment-select exits 4 when the frame index is missing", "[segment]")
{
    const SyntheticSegmentTree tree = MakeSyntheticSegmentTree("cli_missing");
    fs::remove(tree.frameIndexPath);
    const Invocation invocation = Invoke({"--config", tree.configPath.string()});
    CHECK(invocation.code == 4);
}

TEST_CASE("segment-select help lists the smoke and index options", "[segment]")
{
    const Invocation invocation = Invoke({"--help"});
    CHECK(invocation.code == 0);
    CHECK(invocation.out.find("--frame-index") != std::string::npos);
    CHECK(invocation.out.find("--max-frames") != std::string::npos);
    CHECK(invocation.out.find("--stride") != std::string::npos);
}
