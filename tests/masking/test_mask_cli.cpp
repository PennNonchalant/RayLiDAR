#include "fixture.h"

#include <raylidar/cli/commands_phase2.h>
#include <raylidar/common/io.h>

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace
{
namespace fs = std::filesystem;
using raylidar::ReadJson;
using raylidar::test::MakeSyntheticMaskTree;
using raylidar::test::SyntheticMaskTree;

int RunMaskBuildWith(const std::vector<std::string> &arguments)
{
    std::vector<std::string_view> views;
    views.reserve(arguments.size());
    for (const std::string &argument : arguments)
    {
        views.emplace_back(argument);
    }
    return RunMaskBuild(std::span<const std::string_view>(views.data(), views.size()));
}

std::uintmax_t FileSize(const fs::path &path)
{
    std::error_code error;
    const std::uintmax_t size = fs::file_size(path, error);
    return error ? 0U : size;
}
} // namespace

TEST_CASE("mask-build writes one complete artifact per frame", "[masking]")
{
    const SyntheticMaskTree tree = MakeSyntheticMaskTree("mask_cli_ok");
    CHECK(RunMaskBuildWith({"--config", tree.configPath.string(), "--frames", "0,1"}) == 0);

    const fs::path masks = tree.phaseDir / "masks";
    for (const int frame : tree.frames)
    {
        const std::string name =
            "frame_" + std::string(10U - std::to_string(frame).size(), '0') + std::to_string(frame);
        const fs::path frameDir = masks / name;
        CHECK(fs::exists(frameDir / "points.bin"));
        CHECK(FileSize(frameDir / "points.bin") == tree.pointCount);
        CHECK(fs::exists(frameDir / "image_00_dynamic.png"));
        CHECK(fs::exists(frameDir / "image_01_dynamic.png"));

        const nlohmann::json mask = ReadJson(frameDir / "mask.json");
        CHECK(mask["frame"].get<int>() == frame);
        CHECK(mask["point_count"].get<std::size_t>() == tree.pointCount);
        CHECK(mask["counts"].contains("dynamic_union"));
        CHECK(mask["counts"].contains("ego_roi"));
        CHECK(mask.contains("residual_dynamic"));
        CHECK(mask["applies_to"].size() == 4U);
        CHECK(mask["counts"].contains("overlap_2d_only"));
        CHECK(mask["counts"].contains("overlap_bbox_only"));
        CHECK(mask["counts"].contains("overlap_both"));
        CHECK(mask["counts"]["overlap_2d_only"].get<std::size_t>() +
                  mask["counts"]["overlap_bbox_only"].get<std::size_t>() +
                  mask["counts"]["overlap_both"].get<std::size_t>() ==
              mask["counts"]["dynamic_union"].get<std::size_t>());
        CHECK(mask["files"].contains("points_sha256"));
    }

    const nlohmann::json manifest = ReadJson(masks / "mask_manifest.json");
    CHECK(manifest.contains("totals"));
    CHECK(manifest.contains("config_sha256"));
    CHECK(manifest["frames"].size() == tree.frames.size());
    CHECK(manifest["totals"]["kept_static"].get<std::size_t>() ==
          tree.frames.size() * tree.pointCount);
}

TEST_CASE("mask-build exits 4 when semantics are missing", "[masking]")
{
    const SyntheticMaskTree tree = MakeSyntheticMaskTree("mask_cli_missing", false);
    CHECK(RunMaskBuildWith({"--config", tree.configPath.string(), "--frames", "0,1"}) == 4);
}

TEST_CASE("mask-build smoke mode does not claim the final manifest", "[masking]")
{
    const SyntheticMaskTree tree = MakeSyntheticMaskTree("mask_cli_smoke");
    CHECK(RunMaskBuildWith({"--config", tree.configPath.string(), "--frames", "0,1",
                            "--max-frames", "1"}) == 0);
    CHECK(fs::exists(tree.phaseDir / "masks" / "mask_manifest.smoke.json"));
    CHECK_FALSE(fs::exists(tree.phaseDir / "masks" / "mask_manifest.json"));
}
