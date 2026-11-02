#pragma once

#include <raylidar/export.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace raylidar::masking
{
/// One `<objectN>` entry from the KITTI-360 3D bounding-box XML.
///
/// `T_W_B` maps the object's local cube (extent `[-0.5, 0.5]` per axis) into
/// world coordinates; its columns carry both rotation and object scale, so the
/// transform is never normalised. `aabbMin` / `aabbMax` are the world-space
/// axis-aligned bounds of the eight transformed corners and exist only as a
/// cheap prefilter; `T_B_W` is the precomputed inverse used by `PointInBBox`.
struct RAYLIDAR_EXPORT DynamicBBox
{
    std::string label;
    int semanticId = 0;
    int timestamp = -1;
    int startFrame = -1;
    int endFrame = -1;
    bool dynamic = false;
    Eigen::Matrix4d T_W_B = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d T_B_W = Eigen::Matrix4d::Identity();
    Eigen::Vector3d aabbMin = Eigen::Vector3d::Zero();
    Eigen::Vector3d aabbMax = Eigen::Vector3d::Zero();
};

/// Parse an OpenCV XML bbox document.
///
/// Every `<objectN>` must contain `<label>`, `<semanticId>`, `<timestamp>`,
/// `<dynamic>`, `<start_frame>`, `<end_frame>`, and a 4x4 `<transform>` with
/// exactly 16 finite values. Documents containing `<!DOCTYPE` or `<!ENTITY` and
/// documents with more than `maxObjects` entries are rejected. A missing or
/// extra transform value is never inferred. Throws `raylidar::IoError` (exit 4).
RAYLIDAR_EXPORT std::vector<DynamicBBox> LoadDynamicBBoxes(const std::filesystem::path &xmlPath,
                                                           std::size_t maxObjects = 200000);

/// True when `p_W` lies inside the transformed unit cube.
///
/// Points outside the world AABB (plus a small epsilon) are rejected before the
/// inverse transform; inside, `q = T_B_W * p_W` and membership requires
/// `|q.x| <= 0.5 && |q.y| <= 0.5 && |q.z| <= 0.5`.
RAYLIDAR_EXPORT bool PointInBBox(const Eigen::Vector3d &p_W, const DynamicBBox &box);

/// Boxes applicable to `frame`: dynamic, `startFrame <= frame <= endFrame`, and
/// `abs(timestamp - frame) <= maxTimeGapFrames`. The order is box index
/// ascending.
RAYLIDAR_EXPORT std::vector<const DynamicBBox *> ApplicableBoxes(const std::vector<DynamicBBox> &boxes, int frame,
                                                                 int maxTimeGapFrames);
} // namespace raylidar::masking
