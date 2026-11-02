#pragma once

#include <raylidar/export.h>

#include <nlohmann/json.hpp>
#include <yaml-cpp/yaml.h>

#include <cstddef>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace raylidar
{
/// Error raised for unreadable or invalid input files.
class RAYLIDAR_EXPORT IoError : public std::runtime_error
{
  public:
    explicit IoError(const std::string &what, int exitCode = 4);
    [[nodiscard]] int exitCode() const noexcept;

  private:
    int exitCode_;
};

/// SHA-256 of a file, as lowercase hex. Throws IoError on read failure; never
/// returns a partial hash.
RAYLIDAR_EXPORT std::string Sha256File(const std::filesystem::path &path);

/// SHA-256 of a UTF-8 string, as lowercase hex.
RAYLIDAR_EXPORT std::string Sha256String(std::string_view text);

/// SHA-256 of a byte buffer, as lowercase hex.
RAYLIDAR_EXPORT std::string Sha256Bytes(std::span<const std::byte> bytes);

/// SHA-256 of the canonical JSON encoding (see CanonicalJson).
RAYLIDAR_EXPORT std::string Sha256Json(const nlohmann::json &value);

/// Parse a JSON file. Throws IoError (exit 4) on unreadable/invalid input.
RAYLIDAR_EXPORT nlohmann::json ReadJson(const std::filesystem::path &path);

/// Write UTF-8 JSON with 2-space indentation and a trailing newline, creating
/// parent directories as needed.
RAYLIDAR_EXPORT void WriteJson(const std::filesystem::path &path, const nlohmann::json &value);

/// Recursively rebuild a JSON value so object keys are lexicographically sorted.
RAYLIDAR_EXPORT nlohmann::json CanonicalJson(const nlohmann::json &value);

/// Parse a YAML file. An empty file yields an empty map. Invalid YAML throws
/// IoError (exit 4).
RAYLIDAR_EXPORT YAML::Node ReadYaml(const std::filesystem::path &path);

/// Current UTC time formatted as YYYY-MM-DDTHH:MM:SSZ.
RAYLIDAR_EXPORT std::string UtcNowIso8601();

/// Create a directory (and parents) if needed, returning the path.
RAYLIDAR_EXPORT std::filesystem::path EnsureDir(const std::filesystem::path &path);
} // namespace raylidar
