#pragma once

#include <raylidar/export.h>

#include <nlohmann/json.hpp>

#include <filesystem>
#include <string>
#include <string_view>

namespace raylidar::prereg
{
/// SHA-256 of the exact bytes of the decisions document.
RAYLIDAR_EXPORT std::string Sha256Decisions(const std::filesystem::path &path);

/// Return the value after `**Frozen at (UTC):**` on its line, or an empty
/// string when the marker is absent.
RAYLIDAR_EXPORT std::string ReadFrozenAt(const std::filesystem::path &decisions);

/// Replace the `**Frozen at (UTC):**` line with `**Frozen at (UTC):** <utc>`.
/// Every other byte is preserved; the file is written as UTF-8 with a trailing
/// newline. Throws IoError when the marker is missing.
RAYLIDAR_EXPORT void ReplaceFrozenAt(const std::filesystem::path &decisions,
                                     std::string_view utc);

/// Substitute the timestamp, write the `sha256sum`-format sidecar and its
/// `.json` companion, and return the freeze record. The decisions hash is
/// computed over the file after the timestamp substitution.
///
/// `nowUtc` overrides the wall clock (used by deterministic tests).
RAYLIDAR_EXPORT nlohmann::json FreezeDecisions(const std::filesystem::path &decisions,
                                               const std::filesystem::path &protocol,
                                               const std::filesystem::path &sidecar,
                                               std::string_view nowUtc = {});

/// Recompute the decisions hash and compare it with the first 64 hex
/// characters of the sidecar. Returns
/// `{"ok": bool, "expected": str, "actual": str, "frozen_at_utc": str}`.
/// A missing or malformed sidecar throws IoError (exit 4).
RAYLIDAR_EXPORT nlohmann::json VerifyDecisions(const std::filesystem::path &decisions,
                                               const std::filesystem::path &sidecar);
} // namespace raylidar::prereg
