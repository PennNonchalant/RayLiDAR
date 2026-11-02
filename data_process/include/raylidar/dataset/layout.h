#pragma once

#include <raylidar/export.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace raylidar::dataset
{
/// Which on-disk convention a dataset root follows.
///
/// `Official` is the KITTI-360 release layout; `Repackaged` is the local split
/// tree (`raw/2013_05_28/<seq>/image_XX/data`); `Mixed` means both marker sets
/// were observed; `Unknown` means neither was.
enum class LayoutKind
{
    Official,
    Repackaged,
    Mixed,
    Unknown,
};

/// Human-readable name of a layout kind.
RAYLIDAR_EXPORT std::string_view LayoutKindName(LayoutKind kind);

/// Inventory of one dataset component (calibration, poses, images, ...).
struct RAYLIDAR_EXPORT ComponentStatus
{
    std::string name;
    std::vector<std::string> required;
    std::vector<std::string> present;
    std::vector<std::string> missing;
    std::vector<std::string> rejected;
    std::size_t fileCount = 0;
    std::uintmax_t bytes = 0;
    std::string notes;

    /// True when nothing is missing and nothing was rejected.
    [[nodiscard]] bool Ok() const noexcept;
};

/// Resolves the official KITTI-360 paths for one root and sequence.
class RAYLIDAR_EXPORT DatasetLayout
{
  public:
    std::filesystem::path root;
    std::string sequence;
    std::vector<std::string> cameras;
    LayoutKind kind = LayoutKind::Unknown;

    [[nodiscard]] std::filesystem::path CalibrationDir() const;
    [[nodiscard]] std::filesystem::path PosesDir() const;
    [[nodiscard]] std::filesystem::path ImageDir(std::string_view camera) const;
    [[nodiscard]] std::filesystem::path SemanticsDir(std::string_view camera) const;
    [[nodiscard]] std::filesystem::path LidarDir() const;
    [[nodiscard]] std::filesystem::path Path(std::string_view relative) const;
};

/// Detect the layout convention of `root` for `sequence`.
RAYLIDAR_EXPORT DatasetLayout DetectLayout(const std::filesystem::path &root, std::string_view sequence);

/// Find KITTI-raw calibration files (`calib_cam_to_cam.txt`,
/// `calib_velo_to_cam.txt`, `calib_imu_to_velo.txt`) under `root` (and
/// `root/raw`), never following directory symlinks. Results are deduplicated by
/// canonical path and returned as sorted POSIX paths relative to `root`. The
/// scan is read-only and safe to run on an external root.
RAYLIDAR_EXPORT std::vector<std::string> FindRejectedCalibration(const std::filesystem::path &root, int maxDepth = 3);

/// Build the component inventory in the fixed order
/// calibration, poses, images, semantics, lidar. Missing paths are reported,
/// never thrown.
RAYLIDAR_EXPORT std::vector<ComponentStatus> ComponentInventory(const std::filesystem::path &root,
                                                                std::string_view sequence,
                                                                const std::vector<std::string> &cameras);
} // namespace raylidar::dataset
