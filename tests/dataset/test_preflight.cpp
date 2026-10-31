#include "fixture.h"

#include <raylidar/dataset/preflight.h>

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace
{
namespace fs = std::filesystem;
using raylidar::dataset::ComponentStatus;
using raylidar::dataset::PreflightExitCode;
using raylidar::dataset::PreflightReport;
using raylidar::dataset::PreflightReportJson;
using raylidar::dataset::RenderSummary;
using raylidar::dataset::RunPreflight;
using raylidar::test::DatasetTreeOptions;
using raylidar::test::MakeDatasetTree;
using raylidar::test::MakeTempRoot;

constexpr const char *kSequence = "2013_05_28_drive_0000_sync";
const std::vector<std::string> kCameras = {"image_00", "image_01"};
const std::vector<std::string> kForbidden = {"calib_cam_to_cam.txt", "calib_velo_to_cam.txt",
                                             "calib_imu_to_velo.txt"};

PreflightReport Run(const fs::path &root)
{
    return RunPreflight(root, kSequence, kCameras, kForbidden);
}

const ComponentStatus *FindComponent(const PreflightReport &report, const std::string &name)
{
    const auto found =
        std::find_if(report.components.begin(), report.components.end(),
                     [&name](const ComponentStatus &component) { return component.name == name; });
    return found == report.components.end() ? nullptr : &*found;
}

bool ContainsSubstring(const std::vector<std::string> &values, const std::string &needle)
{
    return std::any_of(values.begin(), values.end(), [&needle](const std::string &value)
                       { return value.find(needle) != std::string::npos; });
}

void WriteRaw(const fs::path &path, std::size_t bytes)
{
    std::error_code error;
    fs::create_directories(path.parent_path(), error);
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    REQUIRE(output.good());
    std::string payload(bytes, '\0');
    output.write(payload.data(), static_cast<std::streamsize>(payload.size()));
}
} // namespace

TEST_CASE("A complete official tree is ready", "[dataset]")
{
    const fs::path root = MakeTempRoot("preflight_complete");
    MakeDatasetTree(root, {});

    const PreflightReport report = Run(root);
    CHECK(report.ok);
    CHECK(report.layout == "Official");
    CHECK(PreflightExitCode(report) == 0);
    CHECK(PreflightExitCode(report, true) == 0);
    CHECK(report.components.size() == 5U);
    CHECK(RenderSummary(report).find("ready: yes") != std::string::npos);
}

TEST_CASE("Missing calibration is blocking", "[dataset]")
{
    const fs::path root = MakeTempRoot("preflight_missing_calibration");
    DatasetTreeOptions options;
    options.missing.insert("calibration");
    MakeDatasetTree(root, options);

    const PreflightReport report = Run(root);
    CHECK_FALSE(report.ok);
    CHECK(PreflightExitCode(report) == 2);
    CHECK(ContainsSubstring(report.blockingMissing, "calibration/calib_cam_to_pose.txt"));
    CHECK(RenderSummary(report).find("BLOCKING MISSING") != std::string::npos);

    const nlohmann::json json = PreflightReportJson(report);
    CHECK(json["schema_version"] == 1);
    CHECK_FALSE(json["blocking_missing"].empty());
    CHECK(json["ok"] == false);
}

TEST_CASE("A tree with only KITTI-raw calibration is rejected", "[dataset]")
{
    const fs::path root = MakeTempRoot("preflight_kitti_raw_only");
    DatasetTreeOptions options;
    options.official = false;
    options.kittiRawCalib = true;
    MakeDatasetTree(root, options);

    const PreflightReport report = Run(root);
    const ComponentStatus *calibration = FindComponent(report, "calibration");
    REQUIRE(calibration != nullptr);

    CHECK_FALSE(report.ok);
    CHECK(PreflightExitCode(report) == 2);
    REQUIRE(calibration->rejected.size() == 3U);
    CHECK(ContainsSubstring(calibration->rejected, "calib_cam_to_cam.txt"));
    CHECK(ContainsSubstring(calibration->rejected, "calib_velo_to_cam.txt"));
    CHECK(ContainsSubstring(calibration->rejected, "calib_imu_to_velo.txt"));
    CHECK_FALSE(ContainsSubstring(calibration->present, "calib_cam_to_cam.txt"));
    CHECK(ContainsSubstring(report.rejected, "calib_cam_to_cam.txt"));

    const std::string summary = RenderSummary(report);
    CHECK(summary.find("KITTI-raw calibration") != std::string::npos);
}

TEST_CASE("Missing semantics is blocking", "[dataset]")
{
    const fs::path root = MakeTempRoot("preflight_missing_semantics");
    DatasetTreeOptions options;
    options.missing.insert("semantics");
    MakeDatasetTree(root, options);

    const PreflightReport report = Run(root);
    CHECK_FALSE(report.ok);
    CHECK(PreflightExitCode(report) == 2);
    CHECK(ContainsSubstring(report.blockingMissing, "data_2d_semantics"));
}

TEST_CASE("Missing images and LiDAR is non-blocking unless strict", "[dataset]")
{
    const fs::path root = MakeTempRoot("preflight_missing_nonblocking");
    DatasetTreeOptions options;
    options.missing.insert("images");
    options.missing.insert("lidar");
    MakeDatasetTree(root, options);

    const PreflightReport report = Run(root);
    CHECK_FALSE(report.ok);
    CHECK(PreflightExitCode(report) == 3);
    CHECK(PreflightExitCode(report, true) == 2);
    CHECK_FALSE(report.nonblockingMissing.empty());
    CHECK(report.blockingMissing.empty());
}

TEST_CASE("A wrong-size image is non-blocking rejected", "[dataset]")
{
    const fs::path root = MakeTempRoot("preflight_wrong_size");
    DatasetTreeOptions options;
    options.imageSize = {1242, 375};
    MakeDatasetTree(root, options);

    const PreflightReport report = Run(root);
    CHECK(PreflightExitCode(report) == 3);
    CHECK(PreflightExitCode(report, true) == 2);
    REQUIRE_FALSE(report.nonblockingRejected.empty());
    CHECK(ContainsSubstring(report.nonblockingRejected, "data_rect"));
    CHECK(ContainsSubstring(report.nonblockingRejected, "1242x375"));

    const std::string summary = RenderSummary(report);
    CHECK(summary.find("NON-BLOCKING REJECTED") != std::string::npos);
    CHECK(summary.find("1242x375") != std::string::npos);
}

TEST_CASE("A malformed LiDAR file is non-blocking rejected", "[dataset]")
{
    const fs::path root = MakeTempRoot("preflight_bad_lidar");
    MakeDatasetTree(root, {});
    WriteRaw(root / "data_3d_raw" / kSequence / "velodyne_points" / "data" / "0000000000.bin",
             17U);

    const PreflightReport report = Run(root);
    CHECK(PreflightExitCode(report) == 3);
    CHECK(PreflightExitCode(report, true) == 2);
    CHECK(ContainsSubstring(report.nonblockingRejected, "0000000000.bin"));
    CHECK(RenderSummary(report).find("NON-BLOCKING REJECTED") != std::string::npos);
}

TEST_CASE("A repackaged tree never falsely passes calibration", "[dataset]")
{
    const fs::path root = MakeTempRoot("preflight_repackaged");
    DatasetTreeOptions options;
    options.official = false;
    options.repackaged = true;
    MakeDatasetTree(root, options);

    const PreflightReport report = Run(root);
    CHECK(report.layout == "Repackaged");
    CHECK_FALSE(report.ok);
    CHECK(PreflightExitCode(report) == 2);

    const ComponentStatus *calibration = FindComponent(report, "calibration");
    REQUIRE(calibration != nullptr);
    CHECK_FALSE(calibration->Ok());
    CHECK_FALSE(calibration->missing.empty());
}

TEST_CASE("Invalid roots are distinguished from empty existing roots", "[dataset]")
{
    SECTION("nonexistent root exits 4")
    {
        const fs::path missing = MakeTempRoot("preflight_nonexistent") / "does-not-exist";
        const PreflightReport report = Run(missing);
        CHECK_FALSE(report.ok);
        CHECK(PreflightExitCode(report) == 4);
    }

    SECTION("empty existing root exits 2")
    {
        const fs::path empty = MakeTempRoot("preflight_empty_root");
        const PreflightReport report = Run(empty);
        CHECK_FALSE(report.ok);
        CHECK(PreflightExitCode(report) == 2);
        CHECK(ContainsSubstring(report.blockingMissing, "calibration/calib_cam_to_pose.txt"));
        CHECK(ContainsSubstring(report.blockingMissing, "poses.txt"));
        CHECK(ContainsSubstring(report.blockingMissing, "data_2d_semantics"));
    }
}

TEST_CASE("External scan roots record KITTI-raw calibration as rejected", "[dataset]")
{
    const fs::path root = MakeTempRoot("preflight_external_main");
    const fs::path scanRoot = MakeTempRoot("preflight_external_scan");
    DatasetTreeOptions scanOptions;
    scanOptions.official = false;
    scanOptions.kittiRawCalib = true;
    MakeDatasetTree(scanRoot, scanOptions);

    const PreflightReport report =
        RunPreflight(root, kSequence, kCameras, kForbidden, {scanRoot});

    REQUIRE(report.externalScans.contains(scanRoot.string()));
    const nlohmann::json &entries = report.externalScans[scanRoot.string()];
    REQUIRE(entries.size() == 3U);
    CHECK(ContainsSubstring(entries.get<std::vector<std::string>>(), "calib_cam_to_cam.txt"));
    CHECK(ContainsSubstring(entries.get<std::vector<std::string>>(), "calib_velo_to_cam.txt"));
    CHECK(ContainsSubstring(entries.get<std::vector<std::string>>(), "calib_imu_to_velo.txt"));

    CHECK(RenderSummary(report).find("EXTERNAL SCAN REJECTIONS") != std::string::npos);
}

TEST_CASE("The JSON report has the documented keys and component order", "[dataset]")
{
    const fs::path root = MakeTempRoot("preflight_json");
    MakeDatasetTree(root, {});

    const nlohmann::json json = PreflightReportJson(Run(root));
    CHECK(json["schema_version"] == 1);
    CHECK(json.contains("blocking_missing"));
    CHECK(json.contains("nonblocking_rejected"));
    CHECK(json.contains("external_scans"));
    CHECK(json.contains("rejected"));
    CHECK(json.contains("inventory"));
    CHECK(json["ok"] == true);

    REQUIRE(json["components"].size() == 5U);
    CHECK(json["components"][0]["name"] == "calibration");
    CHECK(json["components"][1]["name"] == "poses");
    CHECK(json["components"][2]["name"] == "images");
    CHECK(json["components"][3]["name"] == "semantics");
    CHECK(json["components"][4]["name"] == "lidar");
    CHECK(json["components"][0]["file_count"].get<std::size_t>() > 0U);
}
