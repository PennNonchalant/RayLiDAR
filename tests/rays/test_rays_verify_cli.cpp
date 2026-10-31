#include "fixture.h"

#include <raylidar/cli/commands_phase2.h>
#include <raylidar/common/io.h>

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace
{
namespace fs = std::filesystem;
using raylidar::ReadJson;
using raylidar::Sha256File;
using raylidar::WriteJson;
using raylidar::test::MakeSyntheticRaysTree;
using raylidar::test::RunSyntheticGtCache;
using raylidar::test::SyntheticRaysTree;

struct CliFixture
{
    SyntheticRaysTree tree;
    fs::path raysManifest;
    fs::path maskManifest;
    fs::path gtManifest;
    fs::path freezePath;
    fs::path waterfallsPath;
};

int InvokeVerify(const std::vector<std::string> &arguments)
{
    std::vector<std::string_view> views;
    views.reserve(arguments.size());
    for (const std::string &argument : arguments)
    {
        views.emplace_back(argument);
    }
    return ::RunRaysVerify(std::span<const std::string_view>(views.data(), views.size()));
}

int InvokeRaysBuild(const std::vector<std::string> &arguments)
{
    std::vector<std::string_view> views;
    views.reserve(arguments.size());
    for (const std::string &argument : arguments)
    {
        views.emplace_back(argument);
    }
    return ::RunRaysBuild(std::span<const std::string_view>(views.data(), views.size()));
}

std::vector<std::uint8_t> ReadAllBytes(const fs::path &path)
{
    std::ifstream input(path, std::ios::binary);
    return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(input),
                                     std::istreambuf_iterator<char>());
}

void FlipByte(const fs::path &path, const std::size_t offset)
{
    std::vector<std::uint8_t> bytes = ReadAllBytes(path);
    REQUIRE(offset < bytes.size());
    bytes[offset] ^= 0x01U;
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char *>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
}

void WriteSyntheticMaskManifest(const CliFixture &fixture)
{
    nlohmann::json frames = nlohmann::json::array();
    nlohmann::json totals = {{"frames", fixture.tree.frames.size()},
                             {"nonfinite", 0},
                             {"ego_roi", fixture.tree.frames.size()},
                             {"dynamic_union", fixture.tree.frames.size()},
                             {"kept_static", fixture.tree.frames.size() * fixture.tree.keptPerFrame},
                             {"dynamic_2d", 0},
                             {"dynamic_bbox", 0},
                             {"overlap_2d_only", 0},
                             {"overlap_bbox_only", 0},
                             {"overlap_both", 0}};
    for (const int frame : fixture.tree.frames)
    {
        nlohmann::json entry;
        entry["frame"] = frame;
        entry["point_count"] = 4;
        entry["counts"] = {{"nonfinite", 0},
                           {"ego_roi", 1},
                           {"dynamic_union", 1},
                           {"kept_static", fixture.tree.keptPerFrame},
                           {"dynamic_2d", 0},
                           {"dynamic_bbox", 0},
                           {"overlap_2d_only", 0},
                           {"overlap_bbox_only", 0},
                           {"overlap_both", 0}};
        frames.push_back(std::move(entry));
    }

    nlohmann::json document;
    document["schema_version"] = 1;
    document["sequence"] = fixture.tree.sequence;
    document["generated_at_utc"] = "2026-10-31T00:00:00Z";
    document["applies_to"] = {"images", "gt", "projections", "evaluation"};
    document["frames"] = std::move(frames);
    document["totals"] = std::move(totals);
    document["config_sha256"] = Sha256File(fixture.tree.configPath);
    WriteJson(fixture.maskManifest, document);
}

CliFixture MakeCliFixture(const std::string_view name)
{
    CliFixture fixture;
    fixture.tree = MakeSyntheticRaysTree(name);
    REQUIRE(RunSyntheticGtCache(fixture.tree) == 0);
    REQUIRE(InvokeRaysBuild({"--config", fixture.tree.configPath.string(), "--gt-manifest",
                             fixture.tree.gtManifestPath.string(), "--frames", "0,1"}) == 0);

    fixture.raysManifest = fixture.tree.phaseDir / "rays" / "rays_manifest.smoke.json";
    fixture.gtManifest = fixture.tree.gtManifestPath;
    fixture.maskManifest = fixture.tree.phaseDir / "masks" / "mask_manifest.json";
    fixture.freezePath = fixture.tree.phaseDir / "rays" / "RAY_SET_FREEZE.json";
    fixture.waterfallsPath = fixture.tree.phaseDir / "rays" / "waterfalls.json";
    WriteSyntheticMaskManifest(fixture);
    return fixture;
}

std::vector<std::string> WaterfallArgs(const CliFixture &fixture)
{
    return {"waterfall", "--config", fixture.tree.configPath.string(),
            "--rays-manifest", fixture.raysManifest.string(),
            "--mask-manifest", fixture.maskManifest.string(),
            "--gt-manifest", fixture.gtManifest.string()};
}

std::vector<std::string> FreezeArgs(const CliFixture &fixture)
{
    std::vector<std::string> arguments = WaterfallArgs(fixture);
    arguments[0] = "freeze";
    arguments.push_back("--freeze");
    arguments.push_back(fixture.freezePath.string());
    return arguments;
}

std::vector<std::string> VerifyArgs(const CliFixture &fixture)
{
    return {"verify", "--rays-manifest", fixture.raysManifest.string(),
            "--freeze", fixture.freezePath.string()};
}
} // namespace

TEST_CASE("rays-verify waterfall, freeze, and verify all succeed on a frozen fixture", "[rays]")
{
    const CliFixture fixture = MakeCliFixture("rays_verify_ok");
    CHECK(InvokeVerify(WaterfallArgs(fixture)) == 0);
    REQUIRE(fs::exists(fixture.waterfallsPath));
    const nlohmann::json waterfalls = ReadJson(fixture.waterfallsPath);
    REQUIRE(waterfalls["frames"].is_array());
    CHECK(waterfalls["frames"].size() == fixture.tree.frames.size());
    for (const nlohmann::json &frame : waterfalls["frames"])
    {
        CHECK(frame["exact"].get<bool>());
    }
    CHECK(waterfalls["totals"]["exact"].get<bool>());

    CHECK(InvokeVerify(FreezeArgs(fixture)) == 0);
    REQUIRE(fs::exists(fixture.freezePath));
    const nlohmann::json freeze = ReadJson(fixture.freezePath);
    CHECK(freeze["ray_set_sha256"].get<std::string>().size() == 64U);
    CHECK(freeze["manifest_sha256"].get<std::string>().size() == 64U);

    CHECK(InvokeVerify(VerifyArgs(fixture)) == 0);
}

TEST_CASE("rays-verify verify exits 6 after a one-byte mutation", "[rays]")
{
    CliFixture fixture = MakeCliFixture("rays_verify_tamper");
    REQUIRE(InvokeVerify(FreezeArgs(fixture)) == 0);
    REQUIRE(InvokeVerify(VerifyArgs(fixture)) == 0);

    FlipByte(fixture.tree.phaseDir / "rays" / "frame_0000000000.bin", 40U);
    CHECK(InvokeVerify(VerifyArgs(fixture)) == 6);
}

TEST_CASE("rays-verify exits 4 when the freeze or input is missing", "[rays]")
{
    const CliFixture fixture = MakeCliFixture("rays_verify_missing");
    fs::remove(fixture.freezePath);
    CHECK(InvokeVerify(VerifyArgs(fixture)) == 4);

    CHECK(InvokeVerify({"waterfall", "--config", fixture.tree.configPath.string(),
                        "--rays-manifest", (fixture.tree.phaseDir / "rays" / "absent.json").string(),
                        "--mask-manifest", fixture.maskManifest.string(), "--gt-manifest",
                        fixture.gtManifest.string()}) == 4);
}

TEST_CASE("rays-verify waterfall exits 6 on a non-closing synthetic waterfall", "[rays]")
{
    const CliFixture fixture = MakeCliFixture("rays_verify_nonclosing");
    nlohmann::json document = ReadJson(fixture.maskManifest);
    document["frames"][0]["point_count"] = 5;
    WriteJson(fixture.maskManifest, document);
    CHECK(InvokeVerify(WaterfallArgs(fixture)) == 6);
}

TEST_CASE("rays-verify help lists every subcommand and path option", "[rays]")
{
    CHECK(InvokeVerify({"--help"}) == 0);
}
