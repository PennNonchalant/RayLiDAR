#pragma once

#include <raylidar/export.h>

#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace raylidar::rays
{
/// Aggregate freeze record for the frozen GT ray set (D-20, RAY-06).
///
/// `per_frame_sha256` is the SHA-256 of the concatenated lowercase hashes of one
/// frame's ray file and its three mask files. `manifest_sha256` covers the
/// canonical `rays_manifest.json`, which itself carries the per-file hashes and
/// the advertised counts. `ray_set_sha256` is the canonical-JSON hash of the
/// freeze payload (excluding `frozen_at_utc`, so identical inputs freeze
/// identically).
struct RAYLIDAR_EXPORT RaySetFreeze
{
    std::string sequence;
    std::vector<int> window;
    std::string config_sha256;
    std::string code_version;
    std::string frozen_at_utc;
    std::string ray_set_sha256;
    std::string manifest_sha256;
    std::vector<std::pair<int, std::string>> per_frame_sha256;
    std::vector<std::pair<int, std::string>> per_frame_meta_sha256;
    nlohmann::json counts;
};

/// Recompute every per-frame SHA-256 and write `RAY_SET_FREEZE.json`.
///
/// Hashes each per-frame ray file, all three mask files, every
/// `frame_<frame:010d>_meta.json`, and the canonical manifest itself. A manifest
/// per-frame hash that disagrees with the recomputed file hash is a hard failure
/// (`raylidar::IoError`, exit 6). A missing file is exit 4. The freeze file is
/// written only after every hash agrees. The returned record mirrors the written
/// document.
RAYLIDAR_EXPORT RaySetFreeze FreezeRaySet(const std::filesystem::path &raysManifest,
                                          const std::filesystem::path &configPath,
                                          const std::filesystem::path &freezePath, std::string_view codeVersion);

/// Recompute hashes and counts, then compare them with the freeze and manifest.
///
/// Returns `{ok, expected, actual, mismatched_frames, count_mismatches}`. Every
/// per-frame ray/mask/meta hash and the canonical manifest hash are recomputed in
/// this invocation, and `n_rays`, `bucket_counts`, `camera_counts`, and the
/// occlusion counts are re-derived from the files. A missing freeze, manifest,
/// or per-frame file throws `raylidar::IoError` (exit 4); a hash or count
/// disagreement yields `ok = false` (exit 6 at the CLI). A stored boolean is
/// never trusted on its own.
RAYLIDAR_EXPORT nlohmann::json VerifyRaySet(const std::filesystem::path &raysManifest,
                                            const std::filesystem::path &freezePath);
} // namespace raylidar::rays
