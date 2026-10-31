#pragma once

#include <raylidar/export.h>

#include <span>
#include <string_view>

/// Exported command handlers declared here so tests can exercise them directly
/// instead of shelling out to `raylidar_cli`.
RAYLIDAR_EXPORT int RunCheckCalib(std::span<const std::string_view> args);
RAYLIDAR_EXPORT int RunDataset(std::span<const std::string_view> args);
RAYLIDAR_EXPORT int RunPrereg(std::span<const std::string_view> args);
