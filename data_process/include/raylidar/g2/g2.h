#pragma once

#include <raylidar/export.h>

#include <nlohmann/json.hpp>

#include <filesystem>
#include <string>

namespace raylidar::g2
{
/// Aggregate the Phase-2 G2 evidence directly from stored artifacts.
///
/// Every success criterion is recomputed from the files on disk; there is no
/// boolean override and missing evidence is a `fail`. The result has the shape
/// `{schema_version, generated_at_utc, sequence, run_id, decisions_sha256,
/// protocol_sha256, frozen_at_utc, protocol, limitations, checks, pass}`.
///
/// `phaseDir` is the Phase-2 artifact directory (e.g. `artifacts/02`).
/// `configPath` is the frozen Phase-2 YAML; its grandparent directory is
/// treated as the project root, so `docs/decisions.md` and
/// `configs/protocol.yaml` resolve the same way as the `prereg` CLI.
RAYLIDAR_EXPORT nlohmann::json EvaluateG2(const std::filesystem::path &phaseDir,
                                          const std::filesystem::path &configPath);

/// Render the G2 report as Markdown.
///
/// The final line is exactly `G2 PASSED` only when `report["pass"] == true`;
/// otherwise it is `G2 NOT PASSED`.
RAYLIDAR_EXPORT std::string RenderG2Markdown(const nlohmann::json &report);
} // namespace raylidar::g2
