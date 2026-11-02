#pragma once

#include <raylidar/export.h>

#include <Eigen/Core>
#include <nlohmann/json.hpp>

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <utility>
#include <vector>

namespace raylidar::eval
{
/// One closed-form scene hit: ray parameter in metres and primitive id.
struct RAYLIDAR_EXPORT AnalyticHit
{
    double t = 0.0;
    std::uint32_t primitive_id = 0;
};

/// Intersect a ray with a bounded, two-sided plane.
///
/// The plane is `plane_origin + a * plane_u + b * plane_v` with
/// `|a| <= half_u` and `|b| <= half_v`; `plane_u`/`plane_v` must be orthonormal.
/// Returns the ray parameter `t` in `[t_min, t_max]`, or `std::nullopt` for a
/// parallel ray, a point outside the rectangle, or a `t` outside the bounds.
RAYLIDAR_EXPORT std::optional<double> RaycastPlane(const Eigen::Vector3d &origin, const Eigen::Vector3d &dir,
                                                   const Eigen::Vector3d &plane_origin, const Eigen::Vector3d &plane_u,
                                                   const Eigen::Vector3d &plane_v, double half_u, double half_v,
                                                   double t_min, double t_max);

/// Intersect a ray with a sphere.
///
/// Returns the smallest root in `[t_min, t_max]`; a ray starting inside returns
/// the exit root. A negative root is never a hit.
RAYLIDAR_EXPORT std::optional<double> RaycastSphere(const Eigen::Vector3d &origin, const Eigen::Vector3d &dir,
                                                    const Eigen::Vector3d &center, double radius, double t_min,
                                                    double t_max);

/// Intersect a ray with an axis-aligned bounding box (slab method).
///
/// Returns the entry `t` from outside and the exit `t` from inside, honoring
/// `[t_min, t_max]`. A slab miss or a parallel miss is `std::nullopt`.
RAYLIDAR_EXPORT std::optional<double> RaycastAabb(const Eigen::Vector3d &origin, const Eigen::Vector3d &dir,
                                                  const Eigen::Vector3d &bbox_min, const Eigen::Vector3d &bbox_max,
                                                  double t_min, double t_max);

/// Deterministic procedural street parameters (D-06/D-07).
struct RAYLIDAR_EXPORT ProceduralStreetConfig
{
    std::uint64_t seed = 20261101;
    double ground_length_m = 80.0;
    double ground_width_m = 20.0;
    int box_count = 18;
    int thin_box_count = 8;
};

/// A deterministic W-frame street mesh: a ground rectangle plus axis-aligned
/// boxes, of which `thin_box_count` carry one horizontal dimension in
/// `[0.05, 0.15]` m and a height in `[2.5, 4.0]` m.
struct RAYLIDAR_EXPORT ProceduralScene
{
    ProceduralStreetConfig config;
    std::vector<Eigen::Vector3d> vertices;
    std::vector<std::array<std::uint32_t, 3>> indices;
    std::vector<std::array<Eigen::Vector3d, 3>> triangles;
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> boxes;
    Eigen::Vector3d bbox_min = Eigen::Vector3d::Zero();
    Eigen::Vector3d bbox_max = Eigen::Vector3d::Zero();

    /// Ground (id 0) plus one id per box (1..N).
    [[nodiscard]] std::uint64_t primitive_count() const
    {
        return 1U + static_cast<std::uint64_t>(boxes.size());
    }
};

/// Generate the deterministic street scene.
///
/// One explicit `std::mt19937_64` seeded once; no global RNG, no time, no
/// locale. Throws `std::invalid_argument` for non-positive or absurd
/// dimensions/counts.
RAYLIDAR_EXPORT ProceduralScene GenerateProceduralStreet(const ProceduralStreetConfig &config);

/// Nearest closed-form hit over the ground plane (id 0) and every box
/// (id = box index + 1), tie-broken by the smaller primitive id.
///
/// This is a placeholder for metric proof, **not** a BVH or a production ray
/// caster.
RAYLIDAR_EXPORT std::optional<AnalyticHit> RaycastScene(const ProceduralScene &scene, const Eigen::Vector3d &origin,
                                                        const Eigen::Vector3d &dir, double t_min, double t_max);

/// Write the scene as binary little-endian PLY (W frame, metres), atomically
/// through a sibling `.tmp` file.
RAYLIDAR_EXPORT void WriteProceduralPly(const std::filesystem::path &path, const ProceduralScene &scene);

/// Manifest for the written PLY: generator, seed, parameters, counts, bbox, and
/// `sha256 = Sha256File(plyPath)`.
RAYLIDAR_EXPORT nlohmann::json BuildSceneManifest(const ProceduralScene &scene, const std::filesystem::path &plyPath);
} // namespace raylidar::eval
