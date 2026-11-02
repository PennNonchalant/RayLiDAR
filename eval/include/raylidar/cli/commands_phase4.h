#pragma once

#include <raylidar/export.h>

#include <span>
#include <string_view>

/// Phase-4 command handlers. Declared here so later Phase-4 plans can add
/// implementations without editing the shared command header.
///
/// Arguments are those following the command name; the return value is the
/// process exit code.
RAYLIDAR_EXPORT int RunMesh(std::span<const std::string_view> args);
