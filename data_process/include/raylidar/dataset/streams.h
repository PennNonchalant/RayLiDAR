#pragma once

#include <raylidar/export.h>

#include <nlohmann/json.hpp>

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace raylidar::dataset
{
/// Presence of every Phase-2 stream required for one frame.
///
/// `next_timestamp` is tracked separately because the uncurl step needs the
/// current-to-next sweep interval (D-05): frame `f` is only usable when both its
/// own timestamp and the non-empty timestamp line at index `f + 1` exist.
struct RAYLIDAR_EXPORT StreamPresence
{
    bool image_00 = false;
    bool image_01 = false;
    bool velodyne = false;
    bool timestamp = false;
    bool next_timestamp = false;
    bool pose = false;
    bool next_pose = false;
    bool cam0_to_world = false;
    bool semantics_00 = false;
    bool semantics_01 = false;
};

/// One frame in the ledger plus every reason it is unusable.
struct RAYLIDAR_EXPORT FrameRecord
{
    int frame = 0;
    StreamPresence present;
    std::vector<std::string> missing;
};

/// The complete, sorted stream intersection: usable frames and the dropped
/// ledger. `usable` is the only frame source consumed downstream.
struct RAYLIDAR_EXPORT StreamIndex
{
    std::vector<std::string> cameras;
    std::vector<FrameRecord> usable;
    std::vector<FrameRecord> dropped;
};

/// Intersect the on-disk streams of `sequence`.
///
/// Frame ids are enumerated from `lidarRoot/data/{frame:010d}.bin` (never from
/// images). A frame is usable only when it has both camera images, the Velodyne
/// scan, its own timestamp, an exact pose and `cam0_to_world` row, and — when
/// `requireNextPose` is set — an exact pose and non-empty timestamp for
/// `frame + 1`; plus 2D semantics for both cameras when `requireSemantics` is
/// set. Every absent stream (and a Velodyne size that is not a multiple of
/// 16 bytes) becomes a logged drop reason; nothing is silently skipped.
///
/// Throws `raylidar::IoError` (exit 4) when an input directory or file is
/// unreadable, or when the frame count exceeds 1,000,000.
RAYLIDAR_EXPORT StreamIndex BuildStreamIndex(const std::filesystem::path &datasetRoot,
                                             const std::filesystem::path &lidarRoot, std::string_view sequence,
                                             const std::vector<std::string> &cameras, bool requireExactPose,
                                             bool requireNextPose, bool requireSemantics);

/// Serialize the index as the `streams/frame_index.json` document.
RAYLIDAR_EXPORT nlohmann::json StreamIndexToJson(const StreamIndex &index, std::string_view sequence,
                                                 const std::filesystem::path &datasetRoot,
                                                 const std::filesystem::path &lidarRoot);

/// Write the complete presence ledger as CSV. The header is exactly
/// `frame,image_00,image_01,velodyne,timestamp,next_timestamp,pose,next_pose,cam0_to_world,semantics_00,semantics_01`.
/// Rows are ascending by frame id.
RAYLIDAR_EXPORT void WriteStreamIndexCsv(const std::filesystem::path &path, const StreamIndex &index);
} // namespace raylidar::dataset
