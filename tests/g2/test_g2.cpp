#include "fixture.h"

#include <raylidar/common/io.h>
#include <raylidar/g2/g2.h>

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace
{
namespace fs = std::filesystem;
using raylidar::ReadJson;
using raylidar::WriteJson;
using raylidar::test::G2FixtureOptions;
using raylidar::test::MakeSyntheticG2Tree;
using raylidar::test::RebuildG2Freeze;
using raylidar::test::SyntheticG2Tree;
using raylidar::test::WriteG2TextFile;

const std::vector<std::string> kCheckIds = {"prereg", "segment",   "uncurl", "masks", "gt",
                                            "rays",   "waterfall", "hash",   "domain"};

nlohmann::json FindCheck(const nlohmann::json &report, const std::string &id)
{
    for (const nlohmann::json &check : report.at("checks"))
    {
        if (check.value("id", "") == id)
        {
            return check;
        }
    }
    throw std::runtime_error("missing G2 check " + id);
}

void RequireFailed(const nlohmann::json &report, const std::string &id)
{
    const nlohmann::json check = FindCheck(report, id);
    REQUIRE(check.at("status").get<std::string>() == "fail");
}

std::vector<std::uint8_t> ReadBytes(const fs::path &path)
{
    std::ifstream input(path, std::ios::binary);
    REQUIRE(input.good());
    return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

void WriteBytes(const fs::path &path, const std::vector<std::uint8_t> &bytes)
{
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    REQUIRE(output.good());
    output.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

void FlipLastByte(const fs::path &path)
{
    std::vector<std::uint8_t> bytes = ReadBytes(path);
    REQUIRE(!bytes.empty());
    bytes.back() = static_cast<std::uint8_t>(bytes.back() ^ 0xFFU);
    WriteBytes(path, bytes);
}

void AppendText(const fs::path &path, const std::string &text)
{
    std::ofstream output(path, std::ios::binary | std::ios::app);
    REQUIRE(output.good());
    output << text;
}
} // namespace

TEST_CASE("G2 passes on a complete synthetic tree", "[g2]")
{
    const SyntheticG2Tree tree = MakeSyntheticG2Tree("complete");
    const nlohmann::json report = raylidar::g2::EvaluateG2(tree.phaseDir, tree.configPath);

    REQUIRE(report.at("pass").get<bool>());
    for (const std::string &id : kCheckIds)
    {
        const nlohmann::json check = FindCheck(report, id);
        REQUIRE(check.at("status").get<std::string>() == "pass");
    }
    REQUIRE(report.at("run_id").get<std::string>() == "p2-60f-step10");
    REQUIRE(report.at("protocol").at("occlusion_source").get<std::string>() == "GT_only");
    REQUIRE(report.at("protocol").at("lidar_origin_beam_protocol").get<std::string>() == "not_in_phase_2");
    REQUIRE(report.at("limitations").is_array());
}

TEST_CASE("G2 fails and names each missing evidence item", "[g2]")
{
    struct Case
    {
        std::string id;
        std::string relative;
    };
    const std::vector<Case> cases = {
        {"segment", "segment/segment_manifest.json"}, {"uncurl", "uncurl/uncurl_report.json"},
        {"masks", "masks/mask_manifest.json"},        {"gt", "gt/gt_manifest.json"},
        {"rays", "rays/rays_manifest.json"},          {"waterfall", "rays/waterfalls.json"},
        {"hash", "rays/RAY_SET_FREEZE.json"},
    };

    for (const Case &item : cases)
    {
        const SyntheticG2Tree tree = MakeSyntheticG2Tree("missing_" + item.id);
        std::error_code error;
        fs::remove(tree.phaseDir / item.relative, error);
        REQUIRE_FALSE(error);
        const nlohmann::json report = raylidar::g2::EvaluateG2(tree.phaseDir, tree.configPath);
        REQUIRE_FALSE(report.at("pass").get<bool>());
        RequireFailed(report, item.id);
    }

    const SyntheticG2Tree tree = MakeSyntheticG2Tree("missing_prereg");
    std::error_code error;
    fs::remove(tree.sidecarPath, error);
    REQUIRE_FALSE(error);
    const nlohmann::json report = raylidar::g2::EvaluateG2(tree.phaseDir, tree.configPath);
    REQUIRE_FALSE(report.at("pass").get<bool>());
    RequireFailed(report, "prereg");
}

TEST_CASE("G2 detects a mutated ray or meta file at the hash check", "[g2]")
{
    {
        const SyntheticG2Tree tree = MakeSyntheticG2Tree("mutated_ray");
        FlipLastByte(tree.phaseDir / "rays" / "frame_0000004615.bin");
        const nlohmann::json report = raylidar::g2::EvaluateG2(tree.phaseDir, tree.configPath);
        REQUIRE_FALSE(report.at("pass").get<bool>());
        RequireFailed(report, "hash");
    }
    {
        const SyntheticG2Tree tree = MakeSyntheticG2Tree("mutated_meta");
        FlipLastByte(tree.phaseDir / "rays" / "frame_0000004615_meta.json");
        const nlohmann::json report = raylidar::g2::EvaluateG2(tree.phaseDir, tree.configPath);
        REQUIRE_FALSE(report.at("pass").get<bool>());
        RequireFailed(report, "hash");
    }
}

TEST_CASE("G2 detects a waterfall frame that does not close", "[g2]")
{
    const SyntheticG2Tree tree = MakeSyntheticG2Tree("bad_waterfall");
    const fs::path path = tree.phaseDir / "rays" / "waterfalls.json";
    nlohmann::json document = ReadJson(path);
    document.at("frames").at(0)["valid"] = document.at("frames").at(0).at("valid").get<std::int64_t>() + 1;
    WriteJson(path, document);

    const nlohmann::json report = raylidar::g2::EvaluateG2(tree.phaseDir, tree.configPath);
    REQUIRE_FALSE(report.at("pass").get<bool>());
    RequireFailed(report, "waterfall");
}

TEST_CASE("G2 checks all three mask families against the ray count", "[g2]")
{
    const SyntheticG2Tree tree = MakeSyntheticG2Tree("short_mask");
    const fs::path path = tree.phaseDir / "rays" / "primary_mask_0000004615.bin";
    std::vector<std::uint8_t> bytes = ReadBytes(path);
    REQUIRE(!bytes.empty());
    bytes.pop_back();
    WriteBytes(path, bytes);

    const nlohmann::json report = raylidar::g2::EvaluateG2(tree.phaseDir, tree.configPath);
    REQUIRE_FALSE(report.at("pass").get<bool>());
    RequireFailed(report, "rays");
}

TEST_CASE("G2 rejects a stale prereg sidecar and a non-60 protocol", "[g2]")
{
    {
        const SyntheticG2Tree tree = MakeSyntheticG2Tree("stale_prereg");
        AppendText(tree.decisionsPath, "\nchanged after freeze\n");
        const nlohmann::json report = raylidar::g2::EvaluateG2(tree.phaseDir, tree.configPath);
        REQUIRE_FALSE(report.at("pass").get<bool>());
        RequireFailed(report, "prereg");
    }
    {
        G2FixtureOptions options;
        options.protocolWindowFrames = 120;
        const SyntheticG2Tree tree = MakeSyntheticG2Tree("stale_protocol", options);
        const nlohmann::json report = raylidar::g2::EvaluateG2(tree.phaseDir, tree.configPath);
        REQUIRE_FALSE(report.at("pass").get<bool>());
        RequireFailed(report, "prereg");
    }
}

TEST_CASE("G2 rejects too few rays, a missing image_01, and a non-60 window", "[g2]")
{
    {
        G2FixtureOptions options;
        options.minValidRays = 20000;
        options.raysPerFrame = std::vector<int>(options.frameCount, 20000);
        options.raysPerFrame[7] = 19999;
        const SyntheticG2Tree tree = MakeSyntheticG2Tree("rays_19999", options);
        const nlohmann::json report = raylidar::g2::EvaluateG2(tree.phaseDir, tree.configPath);
        REQUIRE_FALSE(report.at("pass").get<bool>());
        RequireFailed(report, "rays");
    }
    {
        G2FixtureOptions options;
        options.image01 = false;
        const SyntheticG2Tree tree = MakeSyntheticG2Tree("no_image01", options);
        const nlohmann::json report = raylidar::g2::EvaluateG2(tree.phaseDir, tree.configPath);
        REQUIRE_FALSE(report.at("pass").get<bool>());
        RequireFailed(report, "rays");
    }
    {
        G2FixtureOptions options;
        options.frameCount = 59;
        const SyntheticG2Tree tree = MakeSyntheticG2Tree("short_window", options);
        const nlohmann::json report = raylidar::g2::EvaluateG2(tree.phaseDir, tree.configPath);
        REQUIRE_FALSE(report.at("pass").get<bool>());
        RequireFailed(report, "segment");
    }
}

TEST_CASE("G2 markdown ends with the literal pass marker", "[g2]")
{
    const SyntheticG2Tree good = MakeSyntheticG2Tree("markdown_pass");
    nlohmann::json passReport = raylidar::g2::EvaluateG2(good.phaseDir, good.configPath);
    REQUIRE(raylidar::g2::RenderG2Markdown(passReport).find("G2 PASSED") != std::string::npos);

    passReport["pass"] = false;
    const std::string failed = raylidar::g2::RenderG2Markdown(passReport);
    REQUIRE(failed.find("G2 NOT PASSED") != std::string::npos);
    REQUIRE(failed.rfind("G2 PASSED") == std::string::npos);
}
