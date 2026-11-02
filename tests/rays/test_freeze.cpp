#include "fixture.h"

#include <raylidar/cli/commands_phase2.h>
#include <raylidar/common/io.h>
#include <raylidar/rays/freeze.h>

#include <catch2/catch_test_macros.hpp>

#include <cctype>
#include <cstddef>
#include <cstdint>
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
using raylidar::IoError;
using raylidar::ReadJson;
using raylidar::Sha256File;
using raylidar::WriteJson;
using raylidar::rays::FreezeRaySet;
using raylidar::rays::RaySetFreeze;
using raylidar::rays::VerifyRaySet;
using raylidar::test::MakeSyntheticRaysTree;
using raylidar::test::RunSyntheticGtCache;
using raylidar::test::SyntheticRaysTree;

struct FrozenFixture
{
    SyntheticRaysTree tree;
    fs::path manifestPath;
    fs::path freezePath;
    fs::path raysDir;
    RaySetFreeze freeze;
};

using CommandFn = int (*)(std::span<const std::string_view>);

int Invoke(const CommandFn fn, const std::vector<std::string> &arguments)
{
    std::vector<std::string_view> views;
    views.reserve(arguments.size());
    for (const std::string &argument : arguments)
    {
        views.emplace_back(argument);
    }
    return fn(std::span<const std::string_view>(views.data(), views.size()));
}

std::vector<std::uint8_t> ReadAllBytes(const fs::path &path)
{
    std::ifstream input(path, std::ios::binary);
    return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

void WriteAllBytes(const fs::path &path, const std::vector<std::uint8_t> &bytes)
{
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

void FlipByte(const fs::path &path, const std::size_t offset)
{
    std::vector<std::uint8_t> bytes = ReadAllBytes(path);
    REQUIRE(offset < bytes.size());
    bytes[offset] ^= 0x01U;
    WriteAllBytes(path, bytes);
}

FrozenFixture MakeFrozenFixture(const std::string_view name)
{
    FrozenFixture fixture;
    fixture.tree = MakeSyntheticRaysTree(name);
    REQUIRE(RunSyntheticGtCache(fixture.tree) == 0);
    REQUIRE(Invoke(&::RunRaysBuild, {"--config", fixture.tree.configPath.string(), "--gt-manifest",
                                     fixture.tree.gtManifestPath.string(), "--frames", "0,1"}) == 0);

    fixture.raysDir = fixture.tree.phaseDir / "rays";
    const fs::path smoke = fixture.raysDir / "rays_manifest.smoke.json";
    REQUIRE(fs::exists(smoke));
    fixture.manifestPath = fixture.raysDir / "rays_manifest.json";
    fs::copy_file(smoke, fixture.manifestPath, fs::copy_options::overwrite_existing);
    fixture.freezePath = fixture.raysDir / "RAY_SET_FREEZE.json";
    fixture.freeze =
        FreezeRaySet(fixture.manifestPath, fixture.tree.configPath, fixture.freezePath, "test-code-version");
    return fixture;
}

bool IsHexDigest(const std::string &value)
{
    if (value.size() != 64U)
    {
        return false;
    }
    for (const char character : value)
    {
        if (std::isxdigit(static_cast<unsigned char>(character)) == 0)
        {
            return false;
        }
    }
    return true;
}
} // namespace

TEST_CASE("FreezeRaySet hashes rays, masks, meta files, and the canonical manifest", "[rays]")
{
    const FrozenFixture fixture = MakeFrozenFixture("freeze_ok");
    CHECK(fixture.freeze.sequence == fixture.tree.sequence);
    CHECK(fixture.freeze.window.size() == fixture.tree.frames.size());
    CHECK(IsHexDigest(fixture.freeze.ray_set_sha256));
    CHECK(IsHexDigest(fixture.freeze.manifest_sha256));
    CHECK(fixture.freeze.manifest_sha256 == Sha256File(fixture.manifestPath));
    CHECK(fixture.freeze.per_frame_sha256.size() == fixture.tree.frames.size());
    CHECK(fixture.freeze.per_frame_meta_sha256.size() == fixture.tree.frames.size());
    for (const auto &[frame, hash] : fixture.freeze.per_frame_sha256)
    {
        CHECK(frame >= 0);
        CHECK(IsHexDigest(hash));
    }

    REQUIRE(fs::exists(fixture.freezePath));
    const nlohmann::json document = ReadJson(fixture.freezePath);
    CHECK(document["ray_set_sha256"].get<std::string>() == fixture.freeze.ray_set_sha256);
    CHECK(document["manifest_sha256"].get<std::string>() == fixture.freeze.manifest_sha256);
    CHECK(document["code_version"].get<std::string>() == "test-code-version");
    CHECK(document["counts"]["n_rays"].get<std::size_t>() == fixture.tree.frames.size() * fixture.tree.keptPerFrame);
}

TEST_CASE("two freezes over identical inputs produce the same ray_set_sha256", "[rays]")
{
    const FrozenFixture fixture = MakeFrozenFixture("freeze_deterministic");
    const RaySetFreeze second =
        FreezeRaySet(fixture.manifestPath, fixture.tree.configPath, fixture.freezePath, "test-code-version");
    CHECK(second.ray_set_sha256 == fixture.freeze.ray_set_sha256);
    CHECK(second.manifest_sha256 == fixture.freeze.manifest_sha256);
    CHECK(second.per_frame_sha256 == fixture.freeze.per_frame_sha256);
}

TEST_CASE("a one-byte mutation in a ray file, mask, meta file, or manifest is detected", "[rays]")
{
    SECTION("ray file")
    {
        FrozenFixture fixture = MakeFrozenFixture("freeze_tamper_ray");
        FlipByte(fixture.raysDir / "frame_0000000000.bin", 40U);
        const nlohmann::json result = VerifyRaySet(fixture.manifestPath, fixture.freezePath);
        CHECK_FALSE(result["ok"].get<bool>());
        CHECK_FALSE(result["mismatched_frames"].empty());
    }
    SECTION("mask")
    {
        FrozenFixture fixture = MakeFrozenFixture("freeze_tamper_mask");
        FlipByte(fixture.raysDir / "primary_mask_0000000000.bin", 0U);
        const nlohmann::json result = VerifyRaySet(fixture.manifestPath, fixture.freezePath);
        CHECK_FALSE(result["ok"].get<bool>());
        CHECK_FALSE(result["mismatched_frames"].empty());
    }
    SECTION("meta file")
    {
        FrozenFixture fixture = MakeFrozenFixture("freeze_tamper_meta");
        FlipByte(fixture.raysDir / "frame_0000000000_meta.json", 2U);
        const nlohmann::json result = VerifyRaySet(fixture.manifestPath, fixture.freezePath);
        CHECK_FALSE(result["ok"].get<bool>());
        CHECK_FALSE(result["mismatched_frames"].empty());
    }
    SECTION("canonical manifest")
    {
        FrozenFixture fixture = MakeFrozenFixture("freeze_tamper_manifest");
        std::string text;
        {
            std::ifstream input(fixture.manifestPath, std::ios::binary);
            text.assign(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
        }
        const std::size_t position = text.find("\"schema_version\": 1");
        REQUIRE(position != std::string::npos);
        text[position + std::string("\"schema_version\": ").size()] = '2';
        {
            std::ofstream output(fixture.manifestPath, std::ios::binary | std::ios::trunc);
            output << text;
        }
        const nlohmann::json result = VerifyRaySet(fixture.manifestPath, fixture.freezePath);
        CHECK_FALSE(result["ok"].get<bool>());
        CHECK_FALSE(result["count_mismatches"].empty());
    }
}

TEST_CASE("VerifyRaySet re-derives counts and fails even when every hash matches", "[rays]")
{
    FrozenFixture fixture = MakeFrozenFixture("freeze_count_mismatch");
    nlohmann::json manifest = ReadJson(fixture.manifestPath);
    const std::size_t stored = manifest["frames"][0]["n_rays"].get<std::size_t>();
    manifest["frames"][0]["n_rays"] = stored + 1U;
    WriteJson(fixture.manifestPath, manifest);

    // Re-freeze so the manifest hash matches the tampered counts.
    const RaySetFreeze refrozen =
        FreezeRaySet(fixture.manifestPath, fixture.tree.configPath, fixture.freezePath, "test-code-version");
    CHECK(refrozen.manifest_sha256 == Sha256File(fixture.manifestPath));

    const nlohmann::json result = VerifyRaySet(fixture.manifestPath, fixture.freezePath);
    CHECK_FALSE(result["ok"].get<bool>());
    CHECK_FALSE(result["count_mismatches"].empty());
    CHECK(result["count_mismatches"].dump().find("n_rays") != std::string::npos);
}

TEST_CASE("VerifyRaySet exits 4 on a missing freeze or a missing per-frame file", "[rays]")
{
    const FrozenFixture fixture = MakeFrozenFixture("freeze_missing");
    CHECK_THROWS_AS(VerifyRaySet(fixture.manifestPath, fixture.raysDir / "absent_freeze.json"), IoError);
    try
    {
        static_cast<void>(VerifyRaySet(fixture.manifestPath, fixture.raysDir / "absent_freeze.json"));
    }
    catch (const IoError &error)
    {
        CHECK(error.exitCode() == 4);
    }

    fs::remove(fixture.raysDir / "frame_0000000000.bin");
    CHECK_THROWS_AS(VerifyRaySet(fixture.manifestPath, fixture.freezePath), IoError);
}
