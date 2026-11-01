#pragma once

#include <raylidar/export.h>

namespace raylidar::cli
{
/// Entry point for raylidar_cli. Handles --version/--help and dispatches the
/// remaining arguments to the registered subcommand.
RAYLIDAR_EXPORT int Dispatch(int argc, char *argv[]);
} // namespace raylidar::cli
