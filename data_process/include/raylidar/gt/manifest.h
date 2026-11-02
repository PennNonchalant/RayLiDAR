#pragma once

#include <raylidar/export.h>

#include <nlohmann/json.hpp>

#include <filesystem>
#include <string_view>
#include <vector>

namespace raylidar::gt
{
/// Aggregate the per-frame manifest entries into `<phaseDir>/gt/<manifestFileName>`.
///
/// `frames` entries carry `frame`, `count`, `camera_00`, `camera_01`,
/// `camera_unknown`, `sha256`, and an optional `config_sha256`. The first
/// entry's config hash is promoted to the document `config_sha256` and removed
/// from every per-frame object, so the final per-frame schema stays exactly the
/// six recorded fields. Totals sum `frames`, `points`, `camera_00`,
/// `camera_01`, and `camera_unknown`.
RAYLIDAR_EXPORT nlohmann::json BuildGtManifest(const std::filesystem::path &phaseDir, std::string_view sequence,
                                               const std::vector<nlohmann::json> &frames,
                                               std::string_view manifestFileName = "gt_manifest.json");
} // namespace raylidar::gt
