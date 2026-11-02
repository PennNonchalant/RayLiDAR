#pragma once

#include <raylidar/export.h>

#include <raylidar/eval/aggregate.h>

#include <nlohmann/json.hpp>

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <utility>

namespace raylidar::eval
{
/// Frozen metric protocol loaded from `configs/phase3.yaml`.
///
/// `LoadProtocolConfig` rejects any value that differs from
/// `rays::kRayTMinM`, `rays::kRayMinRangeM`, `rays::kRayMaxRangeM`, `kTausM`,
/// `kPrimaryTauM`, the frozen bucket edges, or a sibling `configs/protocol.yaml`
/// with `IoError` (exit 6).
struct RAYLIDAR_EXPORT ProtocolConfig
{
    int schema_version = 1;
    int protocol_version = 1;
    double t_min_m = 0.0;
    double r_min_m = 0.0;
    double t_max_m = 0.0;
    std::array<double, 3> taus_m = {};
    double primary_tau_m = 0.0;
    std::array<std::pair<double, double>, 3> primary_buckets = {};
    std::pair<double, double> secondary_bucket = {};
};

/// Load and validate the frozen protocol block.
///
/// A missing/unreadable file is `IoError` exit 4; any protocol drift from the
/// frozen constants (or a sibling `protocol.yaml`) is `IoError` exit 6.
RAYLIDAR_EXPORT ProtocolConfig LoadProtocolConfig(const std::filesystem::path &path);

/// Provenance recorded in `metrics.json.inputs`.
struct RAYLIDAR_EXPORT MetricsInputs
{
    std::string mode;
    std::string config_sha256;
    std::string protocol_sha256;
    std::string ray_set_sha256;
    std::string mesh_sha256;
    std::string hit_set_sha256;
    std::string scene_generator;
    std::uint64_t scene_seed = 0;
    std::uint64_t scene_box_count = 0;
    std::uint64_t scene_thin_box_count = 0;
    std::string predictor;
    nlohmann::json predictor_parameters = nlohmann::json::object();
};

/// The frozen tidy-CSV header (schema_version 1), without a trailing newline.
RAYLIDAR_EXPORT const char *MetricsCsvHeader();

/// Serialize an `EvalReport` to the schema_version 1 `metrics.json` value.
///
/// Every undefined metric becomes JSON `null`; a non-finite value is rejected
/// with `IoError` exit 4 rather than serialized.
RAYLIDAR_EXPORT nlohmann::json BuildMetricsJson(const EvalReport &report, const MetricsInputs &inputs,
                                                const ProtocolConfig &protocol, std::string_view generated_at_utc);

/// Write `metrics.json` with 2-space indentation and a trailing newline.
RAYLIDAR_EXPORT void WriteMetricsJson(const std::filesystem::path &path, const nlohmann::json &document);

/// Write the tidy CSV: the frozen header plus one row per frame x camera x bucket.
RAYLIDAR_EXPORT void WriteMetricsCsv(const std::filesystem::path &path, const EvalReport &report,
                                     const ProtocolConfig &protocol);

/// Validate a `metrics.json` value against the frozen schema.
///
/// Rejects a missing required key, a malformed SHA-256, a non-finite number, a
/// protocol drift, a missing/duplicate/misordered row, a count that does not sum
/// to `n_valid`, a row without coverage, an undefined metric encoded as `0`, and
/// a missing caveat. Structural/schema errors are exit 4; hash/protocol drift is
/// exit 6.
RAYLIDAR_EXPORT void ValidateMetricsJson(const nlohmann::json &document);

/// Validate a `metrics.csv` against the frozen header and a validated
/// `metrics.json` document. Rejects a wrong header, a missing/duplicate row, a
/// non-integer count, a `nan`/`inf` token, and a row set that disagrees with the
/// JSON. Throws `IoError` exit 4.
RAYLIDAR_EXPORT void ValidateMetricsCsv(const std::filesystem::path &path, const nlohmann::json &document);
} // namespace raylidar::eval
