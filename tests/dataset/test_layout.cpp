#include "fixture.h"

#include <raylidar/dataset/layout.h>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace
{
namespace fs = std::filesystem;
using raylidar::dataset::ComponentInventory;
using raylidar::dataset::ComponentStatus;
using raylidar::dataset::DetectLayout;
using raylidar::dataset::FindRejectedCalibration;
using raylidar::dataset::LayoutKind;

constexpr const char *kSequence = "2013_05_28_drive_0000_sync";
const std::vector<std::string> kCameras = {"image_00", "image_01"};

const ComponentStatus &FindComponent(const std::vector<ComponentStatus> &components,
                                     const std::string &name)
{
    const auto found =
        std::find_if(components.begin(), components.end(),
                     [&name](const ComponentStatus &component) { return component.name == name; });
    REQUIRE(found != components.end());
    return *found;
}

bool ContainsSubstring(const std::vector<std::string> &values, const std::string &needle)
{
    return std::any_of(values.begin(), values.end(), [&needle](const std::string &value)
                       { return value.find(needle) != std::string::npos; });
}

std::string Join(const std::vector<std::string> &values)
{
    std::string joined;
    for (const std::string &value : values)
    {
        joined += value;
        joined += '\n';
    }
    return joined;
}

std::size_t CountEntriesRecursive(const fs::path &root)
{
    std::error_code error;
    if (!fs::is_directory(root, error))
    {
        return 0U;
    }
    std::size_t count = 0U;
    for (const fs::directory_entry &entry : fs::recursive_directory_iterator(root, error))
    {
        if (error)
        {
            break;
        }
        ++count;
    }
    return count;
}

void WriteText(const fs::path &path, const std::string &text)
{
    std::error_code error;
    fs::create_directories(path.parent_path(), error);
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    REQUIRE(output.good());
    output << text;
}
} // namespace

TEST_CASE("DetectLayout recognises the official layout", "[dataset]")
{
    const fs::path root = raylidar::test::MakeTempRoot("layout_official");
    raylidar::test::MakeDatasetTree(root, {});

    const auto layout = DetectLayout(root, kSequence);
    CHECK(layout.kind == LayoutKind::Official);
    CHECK(layout.sequence == kSequence);
    CHECK(layout.CalibrationDir() == root / "calibration");
    CHECK(layout.PosesDir() == root / "data_poses" / kSequence);
    CHECK(layout.ImageDir("image_00") ==
          root / "data_2d_raw" / kSequence / "image_00" / "data_rect");
    CHECK(layout.SemanticsDir("image_01") ==
          root / "data_2d_semantics" / "train" / kSequence / "image_01" / "semantic");
    CHECK(layout.LidarDir() ==
          root / "data_3d_raw" / kSequence / "velodyne_points" / "data");
}

TEST_CASE("DetectLayout recognises the repackaged and unknown layouts", "[dataset]")
{
    SECTION("repackaged")
    {
        const fs::path root = raylidar::test::MakeTempRoot("layout_repackaged");
        raylidar::test::DatasetTreeOptions options;
        options.official = false;
        options.repackaged = true;
        raylidar::test::MakeDatasetTree(root, options);

        CHECK(DetectLayout(root, kSequence).kind == LayoutKind::Repackaged);
    }

    SECTION("unknown")
    {
        const fs::path root = raylidar::test::MakeTempRoot("layout_unknown");
        CHECK(DetectLayout(root, kSequence).kind == LayoutKind::Unknown);
    }
}

TEST_CASE("ComponentInventory has a fixed component order", "[dataset]")
{
    const fs::path root = raylidar::test::MakeTempRoot("inventory_order");
    raylidar::test::MakeDatasetTree(root, {});

    const auto components = ComponentInventory(root, kSequence, kCameras);
    REQUIRE(components.size() == 5U);
    CHECK(components[0].name == "calibration");
    CHECK(components[1].name == "poses");
    CHECK(components[2].name == "images");
    CHECK(components[3].name == "semantics");
    CHECK(components[4].name == "lidar");
    for (const ComponentStatus &component : components)
    {
        CHECK(component.Ok());
        CHECK(component.fileCount > 0U);
    }
}

TEST_CASE("KITTI-raw calibration names are rejected, never present", "[dataset]")
{
    const fs::path root = raylidar::test::MakeTempRoot("inventory_rejected");
    raylidar::test::DatasetTreeOptions options;
    options.kittiRawCalib = true;
    raylidar::test::MakeDatasetTree(root, options);

    const auto components = ComponentInventory(root, kSequence, kCameras);
    const ComponentStatus &calibration = FindComponent(components, "calibration");

    REQUIRE(calibration.rejected.size() == 3U);
    CHECK(ContainsSubstring(calibration.rejected, "calib_cam_to_cam.txt"));
    CHECK(ContainsSubstring(calibration.rejected, "calib_velo_to_cam.txt"));
    CHECK(ContainsSubstring(calibration.rejected, "calib_imu_to_velo.txt"));
    CHECK_FALSE(ContainsSubstring(calibration.present, "calib_cam_to_cam.txt"));
    CHECK_FALSE(calibration.Ok());
}

TEST_CASE("A wrong image size is rejected with the observed size", "[dataset]")
{
    const fs::path root = raylidar::test::MakeTempRoot("inventory_wrong_size");
    raylidar::test::DatasetTreeOptions options;
    options.imageSize = {1242, 375};
    raylidar::test::MakeDatasetTree(root, options);

    const auto components = ComponentInventory(root, kSequence, kCameras);
    const ComponentStatus &images = FindComponent(components, "images");

    REQUIRE_FALSE(images.rejected.empty());
    CHECK(ContainsSubstring(images.rejected, "1242x375"));
    CHECK_FALSE(images.Ok());
}

TEST_CASE("A malformed poses line is rejected", "[dataset]")
{
    const fs::path root = raylidar::test::MakeTempRoot("inventory_bad_poses");
    raylidar::test::MakeDatasetTree(root, {});

    WriteText(root / "data_poses" / kSequence / "poses.txt", "0 1 2 3\n");

    const auto components = ComponentInventory(root, kSequence, kCameras);
    const ComponentStatus &poses = FindComponent(components, "poses");

    REQUIRE_FALSE(poses.rejected.empty());
    CHECK(ContainsSubstring(poses.rejected, "poses.txt"));
    CHECK_FALSE(poses.Ok());
}

TEST_CASE("FindRejectedCalibration scans an external tree read-only", "[dataset]")
{
    const fs::path outer = raylidar::test::MakeTempRoot("external_scan");
    const fs::path external = outer / "kitti";
    raylidar::test::DatasetTreeOptions options;
    options.official = false;
    options.kittiRawCalib = true;
    raylidar::test::MakeDatasetTree(external, options);
    // MakeDatasetTree resets the external root; recreate the outer marker.
    std::error_code error;
    fs::create_directories(outer, error);

    const std::size_t before = CountEntriesRecursive(external);
    const std::vector<std::string> rejected = FindRejectedCalibration(external);
    const std::size_t after = CountEntriesRecursive(external);

    CHECK(before == after);
    REQUIRE(rejected.size() == 3U);
    CHECK(ContainsSubstring(rejected, "calib_cam_to_cam.txt"));
    CHECK(ContainsSubstring(rejected, "calib_velo_to_cam.txt"));
    CHECK(ContainsSubstring(rejected, "calib_imu_to_velo.txt"));
    CHECK(std::is_sorted(rejected.begin(), rejected.end()));
}

TEST_CASE("FindRejectedCalibration does not duplicate root/raw entries", "[dataset]")
{
    const fs::path root = raylidar::test::MakeTempRoot("dedupe_raw");
    raylidar::test::DatasetTreeOptions options;
    options.kittiRawCalib = true;
    raylidar::test::MakeDatasetTree(root, options);

    const std::vector<std::string> rejected = FindRejectedCalibration(root, 3);
    REQUIRE(rejected.size() == 3U);
    CHECK(ContainsSubstring(rejected, "raw/2013_05_28/calib_cam_to_cam.txt"));
}

TEST_CASE("Official calibration extras are allowed and noted", "[dataset]")
{
    const fs::path root = raylidar::test::MakeTempRoot("calibration_extras");
    raylidar::test::MakeDatasetTree(root, {});

    const fs::path calibration = root / "calibration";
    WriteText(calibration / "calib_sick_to_velo.txt", "1 0 0 0 0 1 0 0 0 0 1 0\n");
    WriteText(calibration / "image_02.yaml", "name: image_02\n");
    WriteText(calibration / "image_03.yaml", "name: image_03\n");

    const auto components = ComponentInventory(root, kSequence, kCameras);
    const ComponentStatus &status = FindComponent(components, "calibration");

    CHECK(status.Ok());
    CHECK(status.rejected.empty());
    CHECK(status.notes.find("calib_sick_to_velo.txt") != std::string::npos);
    CHECK(status.notes.find("image_02.yaml") != std::string::npos);
    CHECK(status.notes.find("image_03.yaml") != std::string::npos);
}

TEST_CASE("Unequal image counts are recorded as a note", "[dataset]")
{
    const fs::path root = raylidar::test::MakeTempRoot("unequal_counts");
    raylidar::test::DatasetTreeOptions options;
    options.imageCounts = {2, 3};
    raylidar::test::MakeDatasetTree(root, options);

    const auto components = ComponentInventory(root, kSequence, kCameras);
    const ComponentStatus &images = FindComponent(components, "images");

    CHECK(images.fileCount == 5U);
    CHECK(images.notes.find("differ") != std::string::npos);
}
