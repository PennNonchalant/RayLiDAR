#pragma once

#include <raylidar/export.h>

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace raylidar::dataset
{
/// Error raised while acquiring or installing dataset components.
///
/// Exit contract:
///   4  missing or impossible link source / configuration error
///   7  missing or invalid official component
///   8  unsafe or unusable archive
class RAYLIDAR_EXPORT AcquisitionError : public std::runtime_error
{
public:
    explicit AcquisitionError(const std::string &what, int exitCode = 7);
    [[nodiscard]] int exitCode() const noexcept;

private:
    int exitCode_;
};

/// Outcome of one SafeExtract call.
struct RAYLIDAR_EXPORT ExtractResult
{
    /// Paths (relative to the destination) actually written, sorted.
    std::vector<std::string> extracted;
    /// Archive members that did not match the allowlist and were not written.
    std::size_t skippedMembers = 0;
    /// Bytes written for allowlisted files.
    std::uintmax_t bytes = 0;
    /// Map of archive member name -> destination-relative path for written files.
    std::map<std::string, std::string> mapping;
    /// Declared uncompressed size of every allowlisted regular file.
    std::uintmax_t allowlistedBytes = 0;
};

/// Extract `archive` into `dest`, writing only members whose (normalised) path
/// matches one of `allowPrefixes`.
///
/// Safety properties, all checked before the first write on a seekable archive:
///   * absolute paths, `..` components, symlinks, hardlinks, devices, FIFOs and
///     other special entries are rejected with AcquisitionError(8);
///   * cumulative uncompressed size and member count are capped;
///   * the free-space guard scales the *allowlisted* extraction bytes, not the
///     whole archive;
///   * existing symlinks inside `dest` are never followed.
///
/// `stream = true` performs a single pass and therefore requires
/// `estimatedExtractBytes > 0` for the disk guard.
///
/// `availableBytesOverride` exists so tests can exercise the disk guard without
/// filling a real filesystem; production callers leave it empty.
RAYLIDAR_EXPORT ExtractResult SafeExtract(
    const std::filesystem::path &archive, const std::filesystem::path &dest,
    const std::vector<std::string> &allowPrefixes, bool stream = false,
    std::uintmax_t maxBytes = 120'000'000'000, std::size_t maxMembers = 1'000'000,
    double diskFactor = 2.5, std::uintmax_t estimatedExtractBytes = 0,
    std::optional<std::uintmax_t> availableBytesOverride = std::nullopt);

/// Install the official KITTI-360 subset for `sequence` from `staging` into
/// `root`, then write a readiness manifest to `manifestPath`.
///
/// `policy` is one of `official`, `reuse_local_after_g1`, or `auto`.
/// `linkSources` is keyed by `image_00`, `image_01`, `lidar`,
/// `image_00_timestamps`, and `image_01_timestamps`.
RAYLIDAR_EXPORT nlohmann::json InstallStaging(
    const std::filesystem::path &staging, const std::filesystem::path &root,
    std::string_view sequence, const std::vector<std::string> &cameras,
    std::string_view policy, const std::map<std::string, std::string> &linkSources,
    const std::filesystem::path &manifestPath, bool streamExtract = false,
    std::uintmax_t estimatedExtractBytes = 0);
} // namespace raylidar::dataset
