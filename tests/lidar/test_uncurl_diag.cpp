#include "fixture.h"

#include <raylidar/calib/kitti360.h>
#include <raylidar/cli/commands_phase2.h>
#include <raylidar/common/io.h>
#include <raylidar/lidar/uncurl_diag.h>

#include <catch2/catch_test_macros.hpp>

#include <Eigen/Core>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace
{
namespace fs = std::filesystem;
using raylidar::IoError;
using raylidar::ReadJson;
using raylidar::WriteJson;
using raylidar::calib::Kitti360Calib;
using raylidar::lidar::EvaluateUncurl;
using raylidar::lidar::UncurlDiagnostic;
using raylidar::lidar::UncurlThresholds;
using raylidar::test::MakeSyntheticUncurlTree;
using raylidar::test::SyntheticUncurlTree;
using raylidar::test::WriteSyntheticMask;

Eigen::Matrix4d Translation(const double x)
{
    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
    transform(0, 3) = x;
    return transform;
}

Kitti360Calib SyntheticCalib()
{
    Kitti360Calib calib;
    calib.T_V_C0 = Eigen::Matrix4d::Identity();
    calib.T_I_Ck["image_00"] = Eigen::Matrix4d::Identity();
    return calib;
}

std::map<int, Eigen::Matrix4d> SyntheticPoses()
{
    return {{0, Translation(0.0)}, {1, Translation(1.0)}, {2, Translation(2.0)}};
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
        result.code = RunUncurlCheck(views);
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

std::string ReadAll(const fs::path &path)
{
    std::ifstream input(path);
    REQUIRE(input.good());
    std::ostringstream text;
    text << input.rdbuf();
    return text.str();
}

std::string FirstLine(const fs::path &path)
{
    std::ifstream input(path);
    REQUIRE(input.good());
    std::string line;
    std::getline(input, line);
    return line;
}
} // namespace

TEST_CASE("Masked diagnostic keeps only kept-static points and reports both variants",
          "[lidar]")
{
    const SyntheticUncurlTree tree = MakeSyntheticUncurlTree("diag_masked", true);
    const UncurlDiagnostic diagnostic =
        EvaluateUncurl(tree.frames, tree.datasetRoot, tree.lidarRoot, tree.maskDir,
                       SyntheticCalib(), SyntheticPoses(), UncurlThresholds{});

    CHECK(diagnostic.rawBins.size() == 72U);
    CHECK(diagnostic.uncurledBins.size() == 72U);
    CHECK(diagnostic.uncurledMatchedCount == tree.staticPointCount);
    CHECK(diagnostic.rawMatchedCount == tree.staticPointCount);
    CHECK(diagnostic.uncurledMaxAbsBinMeanM < 0.05);
    CHECK(diagnostic.uncurledSinusoidAmplitudeM < 0.03);
    CHECK(diagnostic.rawPeakToPeakM > diagnostic.uncurledPeakToPeakM);
    CHECK(diagnostic.pass);
    CHECK(diagnostic.failures.empty());
}

TEST_CASE("A dynamic point is excluded from both raw and uncurled variants", "[lidar]")
{
    const SyntheticUncurlTree tree = MakeSyntheticUncurlTree("diag_dynamic", true);

    const fs::path allMaskDir = tree.root / "masks_all";
    for (const int frame : tree.frames)
    {
        WriteSyntheticMask(allMaskDir, frame, tree.totalPointCount, -1);
    }

    const UncurlDiagnostic masked =
        EvaluateUncurl(tree.frames, tree.datasetRoot, tree.lidarRoot, tree.maskDir,
                       SyntheticCalib(), SyntheticPoses(), UncurlThresholds{});
    const UncurlDiagnostic allKept =
        EvaluateUncurl(tree.frames, tree.datasetRoot, tree.lidarRoot, allMaskDir,
                       SyntheticCalib(), SyntheticPoses(), UncurlThresholds{});

    // The dynamic point would match if it were kept, so including it adds
    // exactly one source match per variant.
    CHECK(allKept.uncurledMatchedCount == tree.staticPointCount + 1U);
    CHECK(allKept.rawMatchedCount == tree.staticPointCount + 1U);
    CHECK(masked.uncurledMatchedCount + 1U == allKept.uncurledMatchedCount);
    CHECK(masked.rawMatchedCount + 1U == allKept.rawMatchedCount);
}

TEST_CASE("A failing flatness threshold produces pass=false with reasons", "[lidar]")
{
    const SyntheticUncurlTree tree = MakeSyntheticUncurlTree("diag_fail", true);
    UncurlThresholds thresholds;
    thresholds.maxAbsBinMeanM = -1.0; // deliberately impossible

    const UncurlDiagnostic diagnostic =
        EvaluateUncurl(tree.frames, tree.datasetRoot, tree.lidarRoot, tree.maskDir,
                       SyntheticCalib(), SyntheticPoses(), thresholds);
    CHECK_FALSE(diagnostic.pass);
    CHECK_FALSE(diagnostic.failures.empty());
    CHECK(diagnostic.failures.front().find("max abs bin mean") != std::string::npos);
}

TEST_CASE("A missing mask directory is a hard input error", "[lidar]")
{
    const SyntheticUncurlTree tree = MakeSyntheticUncurlTree("diag_no_mask", false);
    CHECK_THROWS_AS(EvaluateUncurl(tree.frames, tree.datasetRoot, tree.lidarRoot,
                                   tree.root / "missing_masks", SyntheticCalib(),
                                   SyntheticPoses(), UncurlThresholds{}),
                    IoError);
}

TEST_CASE("uncurl-check writes the masked report and azimuth table", "[lidar]")
{
    const SyntheticUncurlTree tree = MakeSyntheticUncurlTree("diag_cli", true);
    const Invocation invocation =
        Invoke({"--config", tree.configPath.string(), "--frames", "0,1", "--mask-dir",
                tree.maskDir.string(), "--out-dir", tree.outDir.string()});

    CHECK(invocation.code == 0);
    CHECK(invocation.out.find("pass=true") != std::string::npos);

    const fs::path reportPath = tree.outDir / "uncurl_report.json";
    const fs::path csvPath = tree.outDir / "azimuth_bins.csv";
    REQUIRE(fs::is_regular_file(reportPath));
    REQUIRE(fs::is_regular_file(csvPath));

    const nlohmann::json report = ReadJson(reportPath);
    CHECK(report["schema_version"] == 1);
    CHECK(report["pass"] == true);
    CHECK(report["masks_required"] == true);
    CHECK(report["frames"].size() == 2U);
    CHECK(report["mask_dir"] == tree.maskDir.string());
    CHECK(report.contains("rawSinusoidAmplitudeM"));
    CHECK(report.contains("uncurledSinusoidAmplitudeM"));
    CHECK(report.contains("rawPeakToPeakM"));
    CHECK(report.contains("uncurledPeakToPeakM"));
    CHECK(report.contains("thresholds"));
    CHECK(report["raw"]["bins"].size() == 72U);
    CHECK(report["uncurled"]["bins"].size() == 72U);
    CHECK(report["raw"]["matched_count"] == tree.staticPointCount);
    CHECK(report["uncurled"]["matched_count"] == tree.staticPointCount);

    CHECK(FirstLine(csvPath) ==
          "variant,azimuth_deg,count,mean_signed_range_error_m");
}

TEST_CASE("uncurl-check exits 6 when the frozen flatness gate fails", "[lidar]")
{
    const SyntheticUncurlTree tree = MakeSyntheticUncurlTree("diag_cli_gate", true);
    std::string config = ReadAll(tree.configPath);
    const std::string from = "max_abs_bin_mean_m: 0.05";
    const std::size_t position = config.find(from);
    REQUIRE(position != std::string::npos);
    config.replace(position, from.size(), "max_abs_bin_mean_m: 1e-12");
    const fs::path failConfig = tree.root / "phase2_fail.yaml";
    raylidar::test::WriteTextFile(failConfig, config);

    const Invocation invocation =
        Invoke({"--config", failConfig.string(), "--frames", "0,1", "--mask-dir",
                tree.maskDir.string(), "--out-dir", tree.outDir.string()});
    CHECK(invocation.code == 6);
    CHECK(invocation.err.find("gate:") != std::string::npos);
}

TEST_CASE("uncurl-check resolves --frame-list and rejects missing masks with exit 4",
          "[lidar]")
{
    const SyntheticUncurlTree tree = MakeSyntheticUncurlTree("diag_cli_list", true);
    const fs::path manifest = tree.root / "manifest.json";
    WriteJson(manifest, {{"selected", {{"frames", {0, 1}}}}});

    const Invocation listed =
        Invoke({"--config", tree.configPath.string(), "--frame-list", manifest.string(),
                "--mask-dir", tree.maskDir.string(), "--out-dir", tree.outDir.string()});
    CHECK(listed.code == 0);

    const Invocation missingMask =
        Invoke({"--config", tree.configPath.string(), "--frames", "0,1", "--mask-dir",
                (tree.root / "missing_masks").string(), "--out-dir",
                (tree.root / "out_missing").string()});
    CHECK(missingMask.code == 4);
    CHECK(missingMask.err.find("mask") != std::string::npos);
}
