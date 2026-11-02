#pragma once

#include <raylidar/export.h>

#include <filesystem>
#include <string_view>

namespace raylidar
{
/// Root for generated artifacts. Honours RAYLIDAR_ARTIFACTS_ROOT; otherwise
/// falls back to the canonical data-disk location.
RAYLIDAR_EXPORT std::filesystem::path ArtifactsRoot();

/// Per-phase artifact directory, e.g. PhaseArtifactsDir(1) == <root>/01.
RAYLIDAR_EXPORT std::filesystem::path PhaseArtifactsDir(int phase = 1);

/// Path to an artifact inside a phase directory.
RAYLIDAR_EXPORT std::filesystem::path PhaseArtifact(int phase, std::string_view relative);
} // namespace raylidar
