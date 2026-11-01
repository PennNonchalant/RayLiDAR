#pragma once

#include <raylidar/dataset/layout.h>
#include <raylidar/export.h>

#include <nlohmann/json.hpp>

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace raylidar::dataset
{
/// Readiness report produced by RunPreflight.
struct RAYLIDAR_EXPORT PreflightReport
{
    int schemaVersion = 1;
    std::string root;
    std::string sequence;
    std::string layout;
    std::vector<ComponentStatus> components;
    nlohmann::json inventory;
    nlohmann::json externalScans;
    bool ok = false;
    std::vector<std::string> blockingMissing;
    std::vector<std::string> nonblockingMissing;
    std::vector<std::string> nonblockingRejected;
    std::vector<std::string> rejected;
    std::string generatedAtUtc;
};

/// Inspect `root` for the required components and build a readiness report.
/// Blocking components are calibration, poses, and semantics; images and LiDAR
/// are non-blocking. When `scanRoots` is non-empty, each is scanned for
/// KITTI-raw calibration and the result is recorded in `externalScans`.
RAYLIDAR_EXPORT PreflightReport RunPreflight(
    const std::filesystem::path &root, std::string_view sequence,
    const std::vector<std::string> &cameras,
    const std::vector<std::string> &forbiddenNames,
    const std::vector<std::filesystem::path> &scanRoots = {});

/// Exit contract:
///   0  ready
///   2  a blocking component is missing/rejected, or (strict) a non-blocking one
///   3  only non-blocking components are missing or rejected (strict = false)
///   4  invalid root (not a directory)
RAYLIDAR_EXPORT int PreflightExitCode(const PreflightReport &report, bool strict = false);

/// Fixed-width human summary, including BLOCKING MISSING, KITTI-raw calibration,
/// and NON-BLOCKING REJECTED sections when applicable.
RAYLIDAR_EXPORT std::string RenderSummary(const PreflightReport &report);

/// Serialize the report with the stable snake_case schema.
RAYLIDAR_EXPORT nlohmann::json PreflightReportJson(const PreflightReport &report);
} // namespace raylidar::dataset
