#pragma once

#include <raylidar/export.h>

#include <Eigen/Core>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace raylidar::rays
{
// Frozen Phase-2 ray-domain constants (D-16). `WriteRayFrame` and the CLI
// validate the YAML against these values; they are never re-derived per frame.
inline constexpr double kRayTMinM = 1.0e-4;
inline constexpr double kRayMinRangeM = 1.5;
inline constexpr double kRayMaxRangeM = 60.0;
inline constexpr int kRayImageMarginPx = 5;
inline constexpr std::uint16_t kRayVersion = 1;
inline constexpr std::uint16_t kRayRecordSize = 64;
inline constexpr std::size_t kRayHeaderSize = 32;

/// Primary/secondary range buckets (D-17).
///
/// `[0,20)`, `[20,40)`, and `[40,50]` are primary; `(50,60]` is secondary.
/// Exactly 40 and exactly 50 both belong to `B40_50`.
enum class Bucket : std::uint8_t
{
    B0_20 = 0,
    B20_40 = 1,
    B40_50 = 2,
    B50_60 = 3,
};

/// Stable bucket name used in the per-frame metadata and manifest
/// (`0_20`, `20_40`, `40_50`, `50_60`).
RAYLIDAR_EXPORT const char *BucketName(Bucket bucket);

/// Bucket for a camera-frame range. Throws `std::invalid_argument` for a
/// non-finite value or a value outside `[0, 60]`.
RAYLIDAR_EXPORT Bucket BucketOf(double r_gt);

/// One frozen GT-defined ray.
///
/// Frames and conventions:
/// - `T_CL` in this module means `T_C←L` (`T_Ck_rect_V`): it maps a point from
///   the LiDAR/`V` frame to the rectified camera frame.
/// - `p_C`, `dir_C`, and `t_gt` are all expressed in the **assigned camera**
///   frame, so `t_gt = ||p_C||` is the camera-origin range, not `||p_L||`.
/// - `ray_id = (uint64(frame) << 32) | uint32(retainedIndex)`.
/// - `bucket` is a `Bucket` value; `occluded` is filled by the GT-only
///   occlusion pass (default `0`).
///
/// The LiDAR-origin beam-pattern secondary protocol (D-23) is **out of scope
/// for Phase 2** and must not be mixed into this primary ray set.
struct RAYLIDAR_EXPORT Ray
{
    std::uint64_t ray_id = 0;
    Eigen::Vector3f p_L = Eigen::Vector3f::Zero();
    Eigen::Vector3f p_C = Eigen::Vector3f::Zero();
    Eigen::Vector3f dir_C = Eigen::Vector3f::Zero();
    float t_gt = 0.0F;
    std::uint8_t bucket = 0;    ///< `Bucket` value.
    std::uint8_t occluded = 0;  ///< 1 when GT-only occlusion flagged the ray.
    std::uint8_t camera = 0;    ///< 0 = image_00, 1 = image_01.
    std::uint16_t semanticId = 0;
};

/// Write one per-frame ray file atomically.
///
/// Header 32 bytes: magic `"RLRY"`, version `u16 = 1`, record size `u16 = 64`,
/// count `u32`, frame `u32`, reserved 16. Record 64 bytes: `ray_id u64`,
/// `dir_C f32[3]`, `t_gt f32`, `p_L f32[3]`, `p_C f32[3]`, `bucket u8`,
/// `occluded u8`, `camera u8`, `semantic_id u16`, `pad[11]`. Little-endian.
RAYLIDAR_EXPORT void WriteRayFrame(const std::filesystem::path &path, int frame,
                                   const std::vector<Ray> &rays);

/// Read one per-frame ray file (T-02-16).
///
/// Rejects a wrong magic/version/record size, an exact length that differs from
/// `32 + count * 64`, non-finite values, a bucket outside 0-3, a camera outside
/// 0/1, a non-unit direction, a non-positive `t_gt`, or a `bucket` that does
/// not match `t_gt` with `raylidar::IoError` (exit 4).
RAYLIDAR_EXPORT std::vector<Ray> ReadRayFrame(const std::filesystem::path &path,
                                              int *frame = nullptr);

/// Write a one-byte-per-ray mask (values 0/1) atomically.
///
/// The three mask families (primary, secondary, primary-unoccluded) are
/// produced by the CLI; this helper only persists them.
RAYLIDAR_EXPORT void WriteMaskFile(const std::filesystem::path &path,
                                   const std::vector<std::uint8_t> &mask);
} // namespace raylidar::rays
