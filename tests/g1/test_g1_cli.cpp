#include "fixture.h"

#include <raylidar/cli/commands.h>
#include <raylidar/common/io.h>

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace
{
namespace fs = std::filesystem;
using raylidar::ReadJson;
using raylidar::Sha256File;
using raylidar::WriteJson;
using raylidar::test::g1fixture::GoodPreflight;
using raylidar::test::g1fixture::GoodPrereg;
using raylidar::test::g1fixture::GoodReadiness;
using raylidar::test::g1fixture::MakeCalibReport;
using raylidar::test::g1fixture::MakeOverlayReport;
using raylidar::test::g1fixture::MakeVerification;

std::atomic<unsigned long> gCounter{0};

fs::path MakeTempDir(std::string_view name)
{
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const unsigned long counter = gCounter.fetch_add(1U);
    const fs::path dir = fs::temp_directory_path() /
                         ("raylidar_g1_cli_" + std::string(name) + "_" +
                          std::to_string(stamp) + "_" + std::to_string(counter));
    std::error_code error;
    fs::remove_all(dir, error);
    fs::create_directories(dir, error);
    REQUIRE_FALSE(error);
    return dir;
}

void WriteFile(const fs::path &path, std::string_view text)
{
    std::error_code error;
    fs::create_directories(path.parent_path(), error);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    REQUIRE(out.good());
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
}

std::string ReadFile(const fs::path &path)
{
    std::ifstream in(path, std::ios::binary);
    REQUIRE(in.good());
    std::ostringstream contents;
    contents << in.rdbuf();
    return contents.str();
}

std::string ConfigText()
{
    return "dataset_root: /data-20T/hybrid-lidar-sim/raw/kitti360\n"
           "sequence: 2013_05_28_drive_0000_sync\n"
           "cameras: [image_00, image_01]\n"
           "assertions:\n"
           "  roundtrip_atol: 1.0e-9\n"
           "  rotation_atol: 2.0e-6\n"
           "  rotation_det_atol: 2.0e-6\n"
           "  cam0_world_atol: 1.0e-3\n"
           "  camera_height_m: [1.4, 1.9]\n"
           "  camera_height_stability_m: 0.1\n"
           "overlay:\n"
           "  min_frames: 3\n"
           "  min_features: 5\n"
           "  max_offset_px: 2.0\n";
}

struct CliFixture
{
    fs::path dir;
    fs::path config;
    fs::path preflight;
    fs::path readiness;
    fs::path calib;
    fs::path overlay;
    fs::path prereg;
    fs::path decisions;
    fs::path sidecar;
    fs::path verification;
    fs::path out;
};

CliFixture MakeFixture(std::string_view name)
{
    CliFixture fixture;
    fixture.dir = MakeTempDir(name);
    fixture.config = fixture.dir / "frames.yaml";
    WriteFile(fixture.config, ConfigText());

    fixture.preflight = fixture.dir / "preflight.json";
    WriteJson(fixture.preflight, GoodPreflight());
    fixture.readiness = fixture.dir / "readiness.json";
    WriteJson(fixture.readiness, GoodReadiness());
    fixture.calib = fixture.dir / "calib_report.json";
    WriteJson(fixture.calib, MakeCalibReport());
    fixture.overlay = fixture.dir / "overlay_report.json";
    WriteJson(fixture.overlay, MakeOverlayReport());
    fixture.prereg = fixture.dir / "decisions.sha256.json";
    WriteJson(fixture.prereg, GoodPrereg());

    fixture.decisions = fixture.dir / "docs" / "decisions.md";
    WriteFile(fixture.decisions,
              "# Phase 1 — Pre-registered Evaluation Decisions\n\n"
              "**Status:** PRE-REGISTERED\n"
              "**Frozen at (UTC):** 2026-10-31T05:35:58Z\n");
    fixture.sidecar = fixture.dir / "decisions.sha256";
    WriteFile(fixture.sidecar,
              Sha256File(fixture.decisions) + "  " + fixture.decisions.string() + "\n");

    fixture.verification = fixture.dir / "overlay_verification.json";
    WriteJson(fixture.verification, MakeVerification());
    fixture.out = fixture.dir / "G1_REPORT";

    // Defaults (notably overlay_verification.json) must resolve inside this
    // fixture, not the real artifacts directory.
    ::setenv("RAYLIDAR_ARTIFACTS_ROOT", fixture.dir.string().c_str(), 1);
    return fixture;
}

std::vector<std::string> BaseArgs(const CliFixture &fixture)
{
    return {"--config",           fixture.config.string(),
            "--calib-report",     fixture.calib.string(),
            "--overlay-report",   fixture.overlay.string(),
            "--readiness",        fixture.readiness.string(),
            "--preflight",        fixture.preflight.string(),
            "--prereg-json",      fixture.prereg.string(),
            "--decisions",        fixture.decisions.string(),
            "--sidecar",          fixture.sidecar.string(),
            "--out",              fixture.out.string()};
}

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
        result.code = RunG1Check(views);
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
} // namespace

TEST_CASE("g1-check reports pending_human (exit 7) without a verification file", "[g1]")
{
    const CliFixture fixture = MakeFixture("pending");
    std::vector<std::string> args = BaseArgs(fixture);
    const fs::path worksheet = fixture.dir / "overlay_measurement_worksheet.md";
    args.insert(args.end(), {"--emit-worksheet", worksheet.string()});

    const Invocation invocation = Invoke(args);
    CHECK(invocation.code == 7);
    CHECK(invocation.out.find("pending_human") != std::string::npos);
    CHECK(invocation.out.find("G1 NOT PASSED") != std::string::npos);

    CHECK(fs::is_regular_file(fixture.out.string() + ".json"));
    CHECK(fs::is_regular_file(fixture.out.string() + ".md"));
    CHECK(fs::is_regular_file(worksheet));

    const nlohmann::json report = ReadJson(fixture.out.string() + ".json");
    CHECK(report["pass"] == false);
    const nlohmann::json *offsets = nullptr;
    for (const nlohmann::json &check : report["checks"])
    {
        if (check["id"] == "overlay_offsets")
        {
            offsets = &check;
        }
    }
    REQUIRE(offsets != nullptr);
    CHECK((*offsets)["status"] == "pending_human");

    const std::string text = ReadFile(worksheet);
    CHECK(text.find("scale_factor") != std::string::npos);
    CHECK(text.find("measured_offset_4x_px") != std::string::npos);
    CHECK(text.find("max_offset_px") != std::string::npos);
    CHECK(text.find("original 1") != std::string::npos);
    CHECK(text.find("feature_1") != std::string::npos);
    CHECK(text.find("feature_5") != std::string::npos);
}

TEST_CASE("g1-check passes (exit 0) with a valid verification file", "[g1]")
{
    const CliFixture fixture = MakeFixture("pass");
    std::vector<std::string> args = BaseArgs(fixture);
    args.insert(args.end(), {"--verification", fixture.verification.string()});

    const Invocation invocation = Invoke(args);
    CHECK(invocation.code == 0);
    CHECK(invocation.out.find("G1 PASSED") != std::string::npos);

    const nlohmann::json report = ReadJson(fixture.out.string() + ".json");
    CHECK(report["pass"] == true);
    CHECK(report["blocking_failures"].empty());
}

TEST_CASE("g1-check --validate-verification returns 0 or 5", "[g1]")
{
    const CliFixture fixture = MakeFixture("validate");
    const std::vector<std::string> args = {"--config", fixture.config.string(),
                                           "--overlay-report", fixture.overlay.string(),
                                           "--validate-verification",
                                           fixture.verification.string()};

    const Invocation valid = Invoke(args);
    CHECK(valid.code == 0);
    CHECK(valid.out.find("verification OK") != std::string::npos);

    nlohmann::json verification = MakeVerification();
    for (nlohmann::json &entry : verification["frames"])
    {
        entry["max_offset_px"] = 2.0;
    }
    WriteJson(fixture.verification, verification);

    const Invocation invalid = Invoke(args);
    CHECK(invalid.code == 5);
    CHECK(invalid.err.find("not <") != std::string::npos);
}

TEST_CASE("g1-check returns 4 for a missing non-optional input", "[g1]")
{
    const CliFixture fixture = MakeFixture("missing");
    std::vector<std::string> args = BaseArgs(fixture);
    args.insert(args.end(),
                {"--calib-report", (fixture.dir / "does_not_exist.json").string()});

    const Invocation invocation = Invoke(args);
    CHECK(invocation.code == 4);
    CHECK(invocation.err.find("missing input") != std::string::npos);
}

TEST_CASE("g1-check --help lists its options", "[g1]")
{
    const Invocation invocation = Invoke({"--help"});
    CHECK(invocation.code == 0);
    for (const char *option : {"--config", "--calib-report", "--overlay-report",
                               "--readiness", "--preflight", "--prereg-json",
                               "--verification", "--out", "--emit-worksheet",
                               "--validate-verification"})
    {
        CHECK(invocation.out.find(option) != std::string::npos);
    }
}
