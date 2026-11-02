#include "fixture.h"

#include <raylidar/cli/commands_phase2.h>
#include <raylidar/common/io.h>
#include <raylidar/rays/rays.h>

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

namespace
{
namespace fs = std::filesystem;
using raylidar::ReadJson;
using raylidar::Sha256File;
using raylidar::test::MakeSyntheticRaysTree;
using raylidar::test::RunSyntheticGtCache;
using raylidar::test::SyntheticRaysTree;

int RunRaysBuildWith(const std::vector<std::string> &arguments)
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
    return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

fs::path FramePath(const fs::path &directory, const std::string &prefix, const int frame)
{
    const std::string name = std::string(10U - std::to_string(frame).size(), '0') + std::to_string(frame);
    return directory / (prefix + name + ".bin");
}
} // namespace

TEST_CASE("rays-build smoke run writes rays, three masks, a meta file, and a manifest", "[rays]")
{
    const SyntheticRaysTree tree = MakeSyntheticRaysTree("rays_cli_ok");
    REQUIRE(RunSyntheticGtCache(tree) == 0);

    CHECK(RunRaysBuildWith({"--config", tree.configPath.string(), "--gt-manifest", tree.gtManifestPath.string(),
                            "--frames", "0,1"}) == 0);

    const fs::path raysDir = tree.phaseDir / "rays";
    const fs::path manifestPath = raysDir / "rays_manifest.smoke.json";
    REQUIRE(fs::exists(manifestPath));
    CHECK_FALSE(fs::exists(raysDir / "rays_manifest.json"));

    const nlohmann::json manifest = ReadJson(manifestPath);
    CHECK(manifest["schema_version"].get<int>() == 1);
    CHECK(manifest["sequence"].get<std::string>() == tree.sequence);
    REQUIRE(manifest["window"].is_array());
    CHECK(manifest["window"].size() == tree.frames.size());
    CHECK(manifest["totals"]["frames"].get<std::size_t>() == tree.frames.size());
    CHECK(manifest["totals"]["n_rays"].get<std::size_t>() == tree.frames.size() * tree.keptPerFrame);
    CHECK(manifest["totals"]["bucket_counts"]["0_20"].get<std::size_t>() == tree.frames.size() * tree.keptPerFrame);
    CHECK(manifest["totals"]["camera_counts"]["image_00"].get<std::size_t>() == tree.frames.size() * tree.keptPerFrame);
    CHECK(manifest["totals"]["camera_counts"]["image_01"].get<std::size_t>() == 0U);
    CHECK(manifest["totals"]["occluded_count"].get<std::size_t>() == 0U);
    CHECK(manifest["totals"]["full_count"].get<std::size_t>() == tree.frames.size() * tree.keptPerFrame);

    REQUIRE(manifest["frames"].size() == tree.frames.size());
    std::size_t summedRays = 0;
    for (const nlohmann::json &entry : manifest["frames"])
    {
        const int frame = entry["frame"].get<int>();
        const std::size_t count = entry["n_rays"].get<std::size_t>();
        CHECK(count == tree.keptPerFrame);
        summedRays += count;
        CHECK(entry["bucket_counts"]["0_20"].get<std::size_t>() == count);
        CHECK(entry["camera_counts"]["image_00"].get<std::size_t>() == count);
        CHECK(entry["occluded_count"].get<std::size_t>() == 0U);
        CHECK(entry["full_count"].get<std::size_t>() == count);

        const fs::path rayPath = FramePath(raysDir, "frame_", frame);
        REQUIRE(fs::exists(rayPath));
        CHECK(fs::file_size(rayPath) == static_cast<std::uintmax_t>(raylidar::rays::kRayHeaderSize + count * 64U));
        CHECK(ReadAllBytes(rayPath).size() == raylidar::rays::kRayHeaderSize + count * 64U);

        const fs::path primaryPath = FramePath(raysDir, "primary_mask_", frame);
        const fs::path secondaryPath = FramePath(raysDir, "secondary_mask_", frame);
        const fs::path unoccludedPath = FramePath(raysDir, "primary_unoccluded_mask_", frame);
        REQUIRE(fs::exists(primaryPath));
        REQUIRE(fs::exists(secondaryPath));
        REQUIRE(fs::exists(unoccludedPath));
        CHECK(fs::file_size(primaryPath) == count);
        CHECK(fs::file_size(secondaryPath) == count);
        CHECK(fs::file_size(unoccludedPath) == count);
        for (const std::uint8_t value : ReadAllBytes(primaryPath))
        {
            CHECK(value == 1U);
        }
        for (const std::uint8_t value : ReadAllBytes(secondaryPath))
        {
            CHECK(value == 0U);
        }
        for (const std::uint8_t value : ReadAllBytes(unoccludedPath))
        {
            CHECK(value == 1U);
        }

        CHECK(entry["sha256"]["ray"].get<std::string>() == Sha256File(rayPath));
        CHECK(entry["sha256"]["primary_mask"].get<std::string>() == Sha256File(primaryPath));

        const fs::path metaPath = raysDir / ("frame_" + std::string(10U - std::to_string(frame).size(), '0') +
                                             std::to_string(frame) + "_meta.json");
        REQUIRE(fs::exists(metaPath));
        const nlohmann::json meta = ReadJson(metaPath);
        CHECK(meta["frame"].get<int>() == frame);
        CHECK(meta["n_rays"].get<std::size_t>() == count);
        CHECK(meta["bucket_counts"]["0_20"].get<std::size_t>() == count);
        CHECK(meta["occluded_excluded_count"].get<std::size_t>() == count);
        CHECK(meta["full_count"].get<std::size_t>() == count);
        CHECK(meta["sha256"]["ray"].get<std::string>() == entry["sha256"]["ray"].get<std::string>());
        CHECK(meta["sha256"]["secondary_mask"].get<std::string>() ==
              entry["sha256"]["secondary_mask"].get<std::string>());
    }
    CHECK(summedRays == manifest["totals"]["n_rays"].get<std::size_t>());
}

TEST_CASE("rays-build exits 4 without a GT manifest", "[rays]")
{
    const SyntheticRaysTree tree = MakeSyntheticRaysTree("rays_cli_missing");
    CHECK(RunRaysBuildWith({"--config", tree.configPath.string(), "--gt-manifest", tree.gtManifestPath.string()}) == 4);
}

TEST_CASE("rays-build rejects a config whose bucket edges drifted", "[rays]")
{
    const SyntheticRaysTree tree = MakeSyntheticRaysTree("rays_cli_edges");
    REQUIRE(RunSyntheticGtCache(tree) == 0);

    std::ifstream input(tree.configPath);
    const std::string original((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    const std::size_t position = original.find("40.0, 50.0");
    REQUIRE(position != std::string::npos);
    std::string drifted = original;
    drifted.replace(position, std::string("40.0, 50.0").size(), "40.0, 49.0");
    const fs::path driftedConfig = tree.root / "phase2_drifted.yaml";
    {
        std::ofstream output(driftedConfig, std::ios::binary | std::ios::trunc);
        output << drifted;
    }

    CHECK(RunRaysBuildWith({"--config", driftedConfig.string(), "--gt-manifest", tree.gtManifestPath.string(),
                            "--frames", "0,1"}) == 4);
}
