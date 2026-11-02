#pragma once

#include <raylidar/export.h>

#include <Eigen/Core>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace raylidar::io
{
// Frozen simulation I/O constants (docs/sim-io.md, schema_version 1).
inline constexpr std::size_t kSimIoHeaderSize = 32;
inline constexpr std::uint16_t kGtRaysVersion = 1;
inline constexpr std::uint16_t kGtRaysRecordSize = 96;
inline constexpr std::uint16_t kHitsVersion = 1;
inline constexpr std::uint16_t kHitsRecordSize = 24;
inline constexpr std::uint32_t kPrimIdUnavailable = 0xFFFFFFFFU;
// Reject a header-declared count above this before any record allocation.
inline constexpr std::uint32_t kMaxRecordCount = 200000000U;

/// One W-frame camera-origin GT ray (`gt_rays.bin`, record size 96).
///
/// `origin_W` is the camera centre for this ray, `dir_W` is a unit direction,
/// `t_gt` is the true range from that origin, `t_min` is the self-hit guard,
/// and `t_max` is the cast bound. All distances are metres in the `W` frame.
struct RAYLIDAR_EXPORT GtRay
{
    std::uint64_t ray_id = 0;
    std::uint32_t frame = 0;
    std::uint16_t camera = 0;
    Eigen::Vector3d origin_W = Eigen::Vector3d::Zero();
    Eigen::Vector3d dir_W = Eigen::Vector3d::Zero();
    double t_gt = 0.0;
    double t_min = 0.0;
    double t_max = 0.0;
};

/// One simulator prediction (`hits.bin`, record size 24).
///
/// A hit carries a finite `t_pred` inside `[t_min, t_max]`. A miss carries
/// `t_pred = +inf` and `prim_id = kPrimIdUnavailable`.
struct RAYLIDAR_EXPORT Hit
{
    std::uint64_t ray_id = 0;
    bool hit = false;
    double t_pred = std::numeric_limits<double>::infinity();
    std::uint32_t prim_id = kPrimIdUnavailable;
};

/// Shape and hash of a validated `gt_rays.bin`.
struct RAYLIDAR_EXPORT GtRaysInfo
{
    std::uint32_t count = 0;
    std::uint16_t version = 0;
    std::uint16_t record_size = 0;
    std::string sha256;
};

/// Shape and hash of a validated `hits.bin`.
struct RAYLIDAR_EXPORT HitsInfo
{
    std::uint32_t count = 0;
    std::uint16_t version = 0;
    std::uint16_t record_size = 0;
    std::string sha256;
};

/// Write a `gt_rays.bin` atomically (sibling `.tmp` + rename).
///
/// Rejects non-finite values, a non-unit direction, `t_min <= 0`, a range
/// outside `[t_min, t_max]`, an unknown camera, a duplicate `ray_id`, or a
/// count above `kMaxRecordCount` with `IoError` (exit 4).
RAYLIDAR_EXPORT void WriteGtRays(const std::filesystem::path &path, std::span<const GtRay> rays);

/// Read and validate a `gt_rays.bin`.
///
/// Rejects a wrong magic/version/record size, a count above
/// `kMaxRecordCount`, an exact length that differs from
/// `32 + count * 96` (short or long), non-finite values, a non-unit `dir_W`,
/// `t_min <= 0`, `t_gt` outside `[t_min, t_max]`, an unknown camera, or a
/// duplicate `ray_id` with `IoError` (exit 4). When `info` is non-null it is
/// filled only after successful validation.
RAYLIDAR_EXPORT std::vector<GtRay> ReadGtRays(const std::filesystem::path &path, GtRaysInfo *info = nullptr);

/// Write a `hits.bin` atomically (sibling `.tmp` + rename).
///
/// Rejects an invalid `hit` value, a hit with a non-positive/non-finite
/// `t_pred`, or a miss without `+inf` and the `0xFFFFFFFF` `prim_id` sentinel
/// with `IoError` (exit 4).
RAYLIDAR_EXPORT void WriteHits(const std::filesystem::path &path, std::span<const Hit> hits);

/// Read and validate a `hits.bin`.
///
/// Rejects a wrong magic/version/record size, a count above
/// `kMaxRecordCount`, an exact length that differs from `32 + count * 24`,
/// an invalid `hit` value, a hit with a non-positive/non-finite `t_pred`, or a
/// miss without `+inf` and the `0xFFFFFFFF` sentinel with `IoError` (exit 4).
RAYLIDAR_EXPORT std::vector<Hit> ReadHits(const std::filesystem::path &path, HitsInfo *info = nullptr);

/// Validate that `hits` is exactly one ordered record per `rays` record.
///
/// Requires equal sizes, identical ordered `ray_id`, and a well-formed hit
/// (`t_pred` in `[ray.t_min, ray.t_max]`) or miss (`+inf`, sentinel
/// `prim_id`). Every violation throws `IoError` (exit 4).
RAYLIDAR_EXPORT void ValidateHitsAgainstRays(std::span<const GtRay> rays, std::span<const Hit> hits);

/// The three mesh container formats frozen by `docs/sim-io.md`.
enum class MeshFormat
{
    Ply,
    Glb,
    Obj
};

/// Structural description of a validated W-frame triangle mesh.
struct RAYLIDAR_EXPORT MeshStats
{
    MeshFormat format = MeshFormat::Ply;
    std::uint64_t vertex_count = 0;
    std::uint64_t triangle_count = 0;
    Eigen::Vector3d bbox_min = Eigen::Vector3d::Zero();
    Eigen::Vector3d bbox_max = Eigen::Vector3d::Zero();
    bool has_normals = false;
    bool has_colors = false;
    bool float64_positions = false;
    std::string frame;
    std::string units;
};

/// Resolve the mesh format from the file extension (`.ply`, `.glb`, `.obj`).
///
/// Throws `IoError` (exit 4) for an unknown extension.
RAYLIDAR_EXPORT MeshFormat FormatFromExtension(const std::filesystem::path &path);

/// Structurally validate a W-frame mesh and return its shape.
///
/// PLY must be binary little-endian with `comment frame W` and
/// `comment units m`; GLB must be glTF 2.0 with W/metre extras, identity node
/// transforms, and float32 POSITION accessors carrying `min`/`max`; OBJ must
/// carry the two required comments and contain only triangle faces with
/// positive 1-based indices. All three require at least one vertex and one
/// triangle and finite coordinates. Every violation throws `IoError` (exit 4).
RAYLIDAR_EXPORT MeshStats ValidateMeshFile(const std::filesystem::path &path);
} // namespace raylidar::io
