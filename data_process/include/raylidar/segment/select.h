#pragma once

#include <raylidar/calib/kitti360.h>
#include <raylidar/export.h>
#include <raylidar/masking/bbox.h>
#include <raylidar/masking/mask.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace raylidar::segment
{
/// Frozen context for scoring and window selection.
///
/// `timestamps`, when non-null, must be the fully parsed
/// `<lidar_root>/timestamps.txt`; a CLI loads it once and avoids re-reading the
/// file for every frame. `mask` carries the frozen D-08/D-09 bounds.
struct RAYLIDAR_EXPORT SegmentInputs
{
    std::string sequence;
    std::vector<std::string> cameras;
    masking::MaskOptions mask;
    int windowFrames = 60;
    int stepFrames = 10;
    int minScenePointsPerFrame = 20000;
    const std::vector<std::int64_t> *timestamps = nullptr;
};

/// Per-frame dynamic score after ego removal.
///
/// `dynamic_2d` and `dynamic_bbox` are non-exclusive source counts.
/// `dynamic_union` is the numerator: points flagged by either source that are
/// not inside the ego ROI. `kept_static` is the denominator partner. The score
/// is `dynamic_union / (dynamic_union + kept_static)`.
struct RAYLIDAR_EXPORT FrameScore
{
    int frame = 0;
    std::uint64_t dynamic_2d = 0;
    std::uint64_t dynamic_bbox = 0;
    std::uint64_t dynamic_union = 0;
    std::uint64_t ego_roi = 0;
    std::uint64_t kept_static = 0;
    std::uint64_t nonfinite = 0;
    double score = 0.0;
    bool usable = true;
    std::string unusable_reason;
};

/// Micro-averaged score of one 60-frame window.
struct RAYLIDAR_EXPORT WindowScore
{
    int start_frame = 0;
    int end_frame = 0;
    std::size_t frame_count = 0;
    std::uint64_t dynamic_points = 0;
    std::uint64_t static_points = 0;
    double score = 0.0;
};

/// Score one frame with the frozen D-03 dynamic union.
///
/// Loads the raw scan, uncurls it with the exact current/next poses and
/// timestamps, loads both semantic images and camera transforms, and calls
/// `masking::BuildMask`. A missing pose or timestamp is a loud `IoError`
/// (exit 4), never a silent fallback. Ego points are excluded from both the
/// numerator and the denominator; the denominator therefore only contains
/// scene points.
RAYLIDAR_EXPORT FrameScore ScoreFrame(int frame, const std::filesystem::path &datasetRoot,
                                      const std::filesystem::path &lidarRoot, const calib::Kitti360Calib &calib,
                                      const std::map<int, Eigen::Matrix4d> &poses,
                                      const std::vector<masking::DynamicBBox> &boxes, const SegmentInputs &inputs);

/// Every complete window of consecutive usable-frame entries at `stepFrames`.
///
/// `scores` must be the ascending usable-frame list. A window is skipped when
/// any contained entry is unusable (for example too few scene points) and the
/// final partial window is never a candidate.
RAYLIDAR_EXPORT std::vector<WindowScore> CandidateWindows(const std::vector<FrameScore> &scores, int windowFrames,
                                                          int stepFrames);

/// Lowest window score; identical scores resolve to the earliest window.
/// Returns `std::nullopt` when no candidate exists.
RAYLIDAR_EXPORT std::optional<WindowScore> SelectWindow(const std::vector<WindowScore> &windows);

/// Build the `segment_manifest.json` document.
///
/// `config` provides the frozen `selection` block (`window_frames`,
/// `step_frames`, `score`, `tie_break`). `selected.frames` is the canonical,
/// ascending list of the window's usable frame ids; it always has exactly
/// `frame_count` entries and equals the matching `per_frame` slice.
RAYLIDAR_EXPORT nlohmann::json SelectionManifest(const std::vector<FrameScore> &frames,
                                                 const std::vector<WindowScore> &windows,
                                                 const std::optional<WindowScore> &selected, std::string_view sequence,
                                                 std::string_view mode, const nlohmann::json &config);
} // namespace raylidar::segment
