#pragma once

#include <raylidar/export.h>

#include <span>
#include <string_view>

/// Phase-2 command handlers. Declared here (rather than in commands.h) so later
/// Phase-2 plans can add implementations without editing the shared header.
///
/// Every handler has the same shape as the Phase-1 handlers: arguments are those
/// following the command name and the return value is the process exit code.
RAYLIDAR_EXPORT int RunFrameIndex(std::span<const std::string_view> args);
RAYLIDAR_EXPORT int RunUncurlCheck(std::span<const std::string_view> args);
RAYLIDAR_EXPORT int RunMaskBuild(std::span<const std::string_view> args);
RAYLIDAR_EXPORT int RunSegmentSelect(std::span<const std::string_view> args);
RAYLIDAR_EXPORT int RunGtCache(std::span<const std::string_view> args);
RAYLIDAR_EXPORT int RunRaysBuild(std::span<const std::string_view> args);
RAYLIDAR_EXPORT int RunRaysVerify(std::span<const std::string_view> args);
RAYLIDAR_EXPORT int RunG2Check(std::span<const std::string_view> args);
