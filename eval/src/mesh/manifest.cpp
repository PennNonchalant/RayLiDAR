#include <raylidar/mesh/manifest.h>

#include <raylidar/common/io.h>
#include <raylidar/io/sim_io.h>

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

namespace raylidar::mesh
{
namespace
{
namespace fs = std::filesystem;

constexpr int kExitInvalidInput = 4;
constexpr int kExitSchemaDrift = 6;
constexpr double kBboxToleranceM = 1.0e-6;

[[noreturn]] void Fail(const std::string &message, const int exitCode = kExitInvalidInput)
{
    throw IoError(message, exitCode);
}

bool IsSha256Hex(const std::string &text)
{
    if (text.size() != 64U)
    {
        return false;
    }
    for (const char character : text)
    {
        const bool digit = character >= '0' && character <= '9';
        const bool lower = character >= 'a' && character <= 'f';
        if (!digit && !lower)
        {
            return false;
        }
    }
    return true;
}

const nlohmann::json &RequireKey(const nlohmann::json &object, const char *key, const std::string &where)
{
    if (!object.is_object())
    {
        Fail(where + " must be an object");
    }
    if (!object.contains(key))
    {
        Fail(where + " is missing '" + key + "'");
    }
    return object.at(key);
}

std::string RequireString(const nlohmann::json &object, const char *key, const std::string &where)
{
    const nlohmann::json &value = RequireKey(object, key, where);
    if (!value.is_string())
    {
        Fail(where + "." + key + " must be a string");
    }
    return value.get<std::string>();
}

std::uint64_t RequireUint(const nlohmann::json &object, const char *key, const std::string &where)
{
    const nlohmann::json &value = RequireKey(object, key, where);
    if (!value.is_number_unsigned() && !value.is_number_integer())
    {
        Fail(where + "." + key + " must be an integer");
    }
    if (value.is_number_integer() && !value.is_number_unsigned() && value.get<std::int64_t>() < 0)
    {
        Fail(where + "." + key + " must not be negative");
    }
    return value.get<std::uint64_t>();
}

double RequireNumber(const nlohmann::json &object, const char *key, const std::string &where)
{
    const nlohmann::json &value = RequireKey(object, key, where);
    if (!value.is_number() || !std::isfinite(value.get<double>()))
    {
        Fail(where + "." + key + " must be a finite number");
    }
    return value.get<double>();
}

bool RequireBool(const nlohmann::json &object, const char *key, const std::string &where)
{
    const nlohmann::json &value = RequireKey(object, key, where);
    if (!value.is_boolean())
    {
        Fail(where + "." + key + " must be a boolean");
    }
    return value.get<bool>();
}

Eigen::Vector3d RequireVector3(const nlohmann::json &object, const char *key, const std::string &where)
{
    const nlohmann::json &value = RequireKey(object, key, where);
    if (!value.is_array() || value.size() != 3U)
    {
        Fail(where + "." + key + " must hold three numbers");
    }
    Eigen::Vector3d result = Eigen::Vector3d::Zero();
    for (std::size_t axis = 0; axis < 3U; ++axis)
    {
        const nlohmann::json &component = value.at(axis);
        if (!component.is_number() || !std::isfinite(component.get<double>()))
        {
            Fail(where + "." + key + " must hold three finite numbers");
        }
        result[static_cast<Eigen::Index>(axis)] = component.get<double>();
    }
    return result;
}

std::string RequireSha256(const nlohmann::json &object, const char *key, const std::string &where)
{
    const std::string value = RequireString(object, key, where);
    if (!IsSha256Hex(value))
    {
        Fail(where + "." + key + " is not a lowercase 64-hex SHA-256");
    }
    return value;
}

bool HasParentTraversal(const fs::path &path)
{
    for (const fs::path &part : path)
    {
        if (part == "..")
        {
            return true;
        }
    }
    return false;
}

fs::path ResolveUnder(const fs::path &runDir, const std::string &relative, const std::string &where)
{
    if (relative.empty())
    {
        Fail(where + " must not be empty");
    }
    const fs::path path(relative);
    if (path.is_absolute() || path.has_root_name() || path.has_root_directory())
    {
        Fail(where + " must be relative to the run directory");
    }
    if (HasParentTraversal(path))
    {
        Fail(where + " must not contain a '..' component");
    }
    return runDir / path;
}

struct FileEntry
{
    std::string relative_path;
    std::string sha256;
    std::uint64_t bytes = 0;
    std::uint64_t vertex_count = 0;
    std::uint64_t triangle_count = 0;
    fs::path absolute_path;
};

FileEntry ReadFileEntry(const nlohmann::json &files, const char *format, const fs::path &runDir)
{
    const std::string where = std::string("mesh_manifest.json.files.") + format;
    const nlohmann::json &entry = RequireKey(files, format, where);
    FileEntry result;
    result.relative_path = RequireString(entry, "path", where);
    result.sha256 = RequireSha256(entry, "sha256", where);
    result.bytes = RequireUint(entry, "bytes", where);
    if (result.bytes == 0U)
    {
        Fail(where + ".bytes must be positive");
    }
    if (entry.contains("format") && entry.at("format").is_string() && entry.at("format").get<std::string>() != format)
    {
        Fail(where + ".format does not name " + format);
    }
    result.vertex_count = RequireUint(entry, "vertex_count", where);
    result.triangle_count = RequireUint(entry, "triangle_count", where);
    result.absolute_path = ResolveUnder(runDir, result.relative_path, where + ".path");
    return result;
}

void RequireCountsMatch(const raylidar::io::MeshStats &stats, const FileEntry &entry, const std::string &where)
{
    if (stats.vertex_count != entry.vertex_count || stats.triangle_count != entry.triangle_count)
    {
        Fail(where + " declares " + std::to_string(entry.vertex_count) + " vertices / " +
                 std::to_string(entry.triangle_count) + " triangles but the file has " +
                 std::to_string(stats.vertex_count) + " / " + std::to_string(stats.triangle_count),
             kExitSchemaDrift);
    }
}

void RequireBboxMatch(const raylidar::io::MeshStats &stats, const Eigen::Vector3d &bboxMin,
                      const Eigen::Vector3d &bboxMax, const std::string &where)
{
    if ((stats.bbox_min - bboxMin).cwiseAbs().maxCoeff() > kBboxToleranceM ||
        (stats.bbox_max - bboxMax).cwiseAbs().maxCoeff() > kBboxToleranceM)
    {
        Fail(where + " bbox does not match the canonical PLY", kExitSchemaDrift);
    }
}

void RequireHashMatches(const fs::path &path, const std::string &expected, const std::string &where,
                        const bool verify_hashes)
{
    if (!verify_hashes)
    {
        return;
    }
    std::error_code error;
    if (!fs::is_regular_file(path, error))
    {
        Fail(where + " is missing or unreadable: " + path.string());
    }
    if (raylidar::Sha256File(path) != expected)
    {
        Fail(where + " does not match the recorded SHA-256: " + path.string(), kExitSchemaDrift);
    }
}
} // namespace

MeshManifestValidation ValidateMeshManifest(const fs::path &manifest_path, const fs::path &run_dir,
                                            const bool verify_hashes)
{
    const std::string where = "mesh_manifest.json";
    const nlohmann::json manifest = raylidar::ReadJson(manifest_path);
    if (!manifest.is_object())
    {
        Fail(where + " must be a JSON object");
    }

    if (RequireUint(manifest, "schema_version", where) != 1U)
    {
        Fail(where + ".schema_version must be 1");
    }
    if (RequireString(manifest, "kind", where) != "mesh")
    {
        Fail(where + ".kind must be 'mesh'");
    }
    if (RequireString(manifest, "frame", where) != "W")
    {
        Fail(where + ".frame must be W");
    }
    if (RequireString(manifest, "units", where) != "m")
    {
        Fail(where + ".units must be m");
    }
    if (RequireString(manifest, "route", where).empty())
    {
        Fail(where + ".route must not be empty");
    }
    static_cast<void>(RequireSha256(manifest, "config_sha256", where));

    const std::uint64_t vertex_count = RequireUint(manifest, "vertex_count", where);
    const std::uint64_t triangle_count = RequireUint(manifest, "triangle_count", where);
    if (vertex_count == 0U || triangle_count == 0U)
    {
        Fail(where + " vertex_count/triangle_count must be positive");
    }
    const Eigen::Vector3d bbox_min = RequireVector3(manifest, "bbox_min_m", where);
    const Eigen::Vector3d bbox_max = RequireVector3(manifest, "bbox_max_m", where);
    for (int axis = 0; axis < 3; ++axis)
    {
        if (bbox_min[axis] > bbox_max[axis])
        {
            Fail(where + ".bbox_min_m exceeds bbox_max_m on an axis");
        }
    }

    const nlohmann::json &components = RequireKey(manifest, "components", where);
    const std::uint64_t component_count = RequireUint(components, "count", where + ".components");
    if (component_count < 1U)
    {
        Fail(where + ".components.count must be at least 1");
    }
    const double largest_fraction = RequireNumber(components, "largest_fraction", where + ".components");
    if (largest_fraction < 0.0 || largest_fraction > 1.0)
    {
        Fail(where + ".components.largest_fraction must be in [0, 1]");
    }

    const nlohmann::json &degenerate = RequireKey(manifest, "degenerate", where);
    static_cast<void>(RequireUint(degenerate, "count", where + ".degenerate"));
    const double degenerate_fraction = RequireNumber(degenerate, "fraction", where + ".degenerate");
    if (degenerate_fraction < 0.0 || degenerate_fraction > 1.0)
    {
        Fail(where + ".degenerate.fraction must be in [0, 1]");
    }

    const nlohmann::json &files = RequireKey(manifest, "files", where);
    const FileEntry ply = ReadFileEntry(files, "ply", run_dir);
    const FileEntry glb = ReadFileEntry(files, "glb", run_dir);
    const FileEntry obj = ReadFileEntry(files, "obj", run_dir);

    if (ply.vertex_count != vertex_count || ply.triangle_count != triangle_count)
    {
        Fail(where + " canonical counts must equal files.ply counts", kExitSchemaDrift);
    }

    for (const auto &[label, entry] :
         {std::pair<std::string, FileEntry>{"ply", ply}, std::pair<std::string, FileEntry>{"glb", glb},
          std::pair<std::string, FileEntry>{"obj", obj}})
    {
        const std::string file_where = where + ".files." + label;
        const raylidar::io::MeshStats stats = raylidar::io::ValidateMeshFile(entry.absolute_path);
        RequireCountsMatch(stats, entry, file_where);
        RequireBboxMatch(stats, bbox_min, bbox_max, file_where);
        RequireHashMatches(entry.absolute_path, entry.sha256, file_where, verify_hashes);
    }

    const nlohmann::json &viewer = RequireKey(manifest, "viewer", where);
    const bool viewer_enabled = RequireBool(viewer, "enabled", where + ".viewer");
    if (viewer_enabled)
    {
        const std::string viewer_where = where + ".viewer";
        const std::string relative = RequireString(viewer, "path", viewer_where);
        const std::string sha256 = RequireSha256(viewer, "sha256", viewer_where);
        const std::uint64_t viewer_vertices = RequireUint(viewer, "vertex_count", viewer_where);
        const std::uint64_t viewer_triangles = RequireUint(viewer, "triangle_count", viewer_where);
        if (viewer_vertices == 0U || viewer_triangles == 0U)
        {
            Fail(viewer_where + " counts must be positive");
        }
        const fs::path viewer_path = ResolveUnder(run_dir, relative, viewer_where + ".path");
        const raylidar::io::MeshStats stats = raylidar::io::ValidateMeshFile(viewer_path);
        FileEntry viewer_entry;
        viewer_entry.vertex_count = viewer_vertices;
        viewer_entry.triangle_count = viewer_triangles;
        RequireCountsMatch(stats, viewer_entry, viewer_where);
        RequireHashMatches(viewer_path, sha256, viewer_where, verify_hashes);
    }

    MeshManifestValidation result;
    result.ok = true;
    result.message = "OK";
    result.vertex_count = vertex_count;
    result.triangle_count = triangle_count;
    result.bbox_min = bbox_min;
    result.bbox_max = bbox_max;
    return result;
}
} // namespace raylidar::mesh
