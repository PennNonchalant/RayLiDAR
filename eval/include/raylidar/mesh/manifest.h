#pragma once

#include <raylidar/export.h>

#include <Eigen/Core>

#include <cstdint>
#include <filesystem>
#include <string>

namespace raylidar::mesh
{
/// The validated shape of a Phase-4 `mesh_manifest.json`.
struct RAYLIDAR_EXPORT MeshManifestValidation
{
    bool ok = false;
    std::string message;
    std::uint64_t vertex_count = 0;
    std::uint64_t triangle_count = 0;
    Eigen::Vector3d bbox_min = Eigen::Vector3d::Zero();
    Eigen::Vector3d bbox_max = Eigen::Vector3d::Zero();
};

/// Validate a Phase-4 mesh manifest and the files it declares.
///
/// Structural/semantic problems (missing keys, an unsafe relative path, a
/// malformed mesh file) throw `IoError` with exit code 4. A declared
/// SHA-256/count/bbox that disagrees with the file on disk throws `IoError`
/// with exit code 6 (`--verify-hashes` recomputes every recorded digest).
///
/// `run_dir` is the root the manifest's relative `files.*.path` entries are
/// resolved against; absolute paths and any `..` segment are rejected.
///
/// On success returns `ok = true` and the canonical PLY counts/bbox.
RAYLIDAR_EXPORT MeshManifestValidation ValidateMeshManifest(const std::filesystem::path &manifest_path,
                                                            const std::filesystem::path &run_dir, bool verify_hashes);
} // namespace raylidar::mesh
