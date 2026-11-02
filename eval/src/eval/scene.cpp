#include <raylidar/eval/scene.h>

#include <raylidar/common/io.h>

#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace raylidar::eval
{
namespace
{
namespace fs = std::filesystem;

constexpr double kParallelEpsilon = 1.0e-12;

void PutF64(std::vector<std::uint8_t> &bytes, const double value)
{
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    for (int shift = 0; shift < 64; shift += 8)
    {
        bytes.push_back(static_cast<std::uint8_t>((bits >> shift) & 0xFFU));
    }
}

void PutI32(std::vector<std::uint8_t> &bytes, const std::int32_t value)
{
    const std::uint32_t bits = static_cast<std::uint32_t>(value);
    for (int shift = 0; shift < 32; shift += 8)
    {
        bytes.push_back(static_cast<std::uint8_t>((bits >> shift) & 0xFFU));
    }
}

void AppendFace(ProceduralScene &scene, const std::uint32_t a, const std::uint32_t b, const std::uint32_t c)
{
    scene.indices.push_back({a, b, c});
    scene.triangles.push_back({scene.vertices[a], scene.vertices[b], scene.vertices[c]});
}

void AppendBox(ProceduralScene &scene, const Eigen::Vector3d &minimum, const Eigen::Vector3d &maximum)
{
    const std::uint32_t base = static_cast<std::uint32_t>(scene.vertices.size());
    const double x0 = minimum.x();
    const double y0 = minimum.y();
    const double z0 = minimum.z();
    const double x1 = maximum.x();
    const double y1 = maximum.y();
    const double z1 = maximum.z();

    scene.vertices.emplace_back(x0, y0, z0); // 0
    scene.vertices.emplace_back(x1, y0, z0); // 1
    scene.vertices.emplace_back(x1, y1, z0); // 2
    scene.vertices.emplace_back(x0, y1, z0); // 3
    scene.vertices.emplace_back(x0, y0, z1); // 4
    scene.vertices.emplace_back(x1, y0, z1); // 5
    scene.vertices.emplace_back(x1, y1, z1); // 6
    scene.vertices.emplace_back(x0, y1, z1); // 7

    // Outward-wound triangles: bottom, top, front, back, left, right.
    AppendFace(scene, base + 0U, base + 2U, base + 1U);
    AppendFace(scene, base + 0U, base + 3U, base + 2U);
    AppendFace(scene, base + 4U, base + 5U, base + 6U);
    AppendFace(scene, base + 4U, base + 6U, base + 7U);
    AppendFace(scene, base + 0U, base + 1U, base + 5U);
    AppendFace(scene, base + 0U, base + 5U, base + 4U);
    AppendFace(scene, base + 3U, base + 7U, base + 6U);
    AppendFace(scene, base + 3U, base + 6U, base + 2U);
    AppendFace(scene, base + 0U, base + 4U, base + 7U);
    AppendFace(scene, base + 0U, base + 7U, base + 3U);
    AppendFace(scene, base + 1U, base + 2U, base + 6U);
    AppendFace(scene, base + 1U, base + 6U, base + 5U);

    scene.boxes.emplace_back(minimum, maximum);
}
} // namespace

std::optional<double> RaycastPlane(const Eigen::Vector3d &origin, const Eigen::Vector3d &dir,
                                   const Eigen::Vector3d &plane_origin, const Eigen::Vector3d &plane_u,
                                   const Eigen::Vector3d &plane_v, const double half_u, const double half_v,
                                   const double t_min, const double t_max)
{
    const Eigen::Vector3d cross = plane_u.cross(plane_v);
    const double length = cross.norm();
    if (!(length > kParallelEpsilon) || !std::isfinite(length))
    {
        return std::nullopt;
    }
    const Eigen::Vector3d normal = cross / length;
    const double denominator = dir.dot(normal);
    if (std::abs(denominator) < kParallelEpsilon)
    {
        return std::nullopt;
    }
    const double t = (plane_origin - origin).dot(normal) / denominator;
    if (!std::isfinite(t) || t < t_min || t > t_max)
    {
        return std::nullopt;
    }
    const Eigen::Vector3d point = origin + t * dir;
    const Eigen::Vector3d offset = point - plane_origin;
    if (std::abs(offset.dot(plane_u)) > half_u || std::abs(offset.dot(plane_v)) > half_v)
    {
        return std::nullopt;
    }
    return t;
}

std::optional<double> RaycastSphere(const Eigen::Vector3d &origin, const Eigen::Vector3d &dir,
                                    const Eigen::Vector3d &center, const double radius, const double t_min,
                                    const double t_max)
{
    if (!(radius > 0.0))
    {
        return std::nullopt;
    }
    const Eigen::Vector3d offset = origin - center;
    const double a = dir.squaredNorm();
    if (!(a > kParallelEpsilon))
    {
        return std::nullopt;
    }
    const double b = 2.0 * offset.dot(dir);
    const double c = offset.squaredNorm() - radius * radius;
    const double discriminant = b * b - 4.0 * a * c;
    if (discriminant < 0.0)
    {
        return std::nullopt;
    }
    const double root = std::sqrt(discriminant);
    const double nearRoot = (-b - root) / (2.0 * a);
    const double farRoot = (-b + root) / (2.0 * a);
    if (nearRoot >= t_min && nearRoot <= t_max)
    {
        return nearRoot;
    }
    if (farRoot >= t_min && farRoot <= t_max)
    {
        return farRoot;
    }
    return std::nullopt;
}

std::optional<double> RaycastAabb(const Eigen::Vector3d &origin, const Eigen::Vector3d &dir,
                                  const Eigen::Vector3d &bbox_min, const Eigen::Vector3d &bbox_max, const double t_min,
                                  const double t_max)
{
    double entry = -std::numeric_limits<double>::infinity();
    double exit = std::numeric_limits<double>::infinity();
    for (int axis = 0; axis < 3; ++axis)
    {
        const double direction = dir[axis];
        const double start = origin[axis];
        const double low = bbox_min[axis];
        const double high = bbox_max[axis];
        if (std::abs(direction) < kParallelEpsilon)
        {
            if (start < low || start > high)
            {
                return std::nullopt;
            }
            continue;
        }
        double t1 = (low - start) / direction;
        double t2 = (high - start) / direction;
        if (t1 > t2)
        {
            std::swap(t1, t2);
        }
        entry = std::max(entry, t1);
        exit = std::min(exit, t2);
        if (entry > exit)
        {
            return std::nullopt;
        }
    }
    if (exit < t_min || entry > t_max)
    {
        return std::nullopt;
    }
    if (entry >= t_min)
    {
        return entry;
    }
    // Origin inside the box (or exactly on a slab before `t_min`): use the exit.
    if (exit >= t_min && exit <= t_max)
    {
        return exit;
    }
    return std::nullopt;
}

ProceduralScene GenerateProceduralStreet(const ProceduralStreetConfig &config)
{
    if (!(config.ground_length_m >= 4.0) || !(config.ground_width_m >= 2.0))
    {
        throw std::invalid_argument("procedural ground must be at least 4 m long and 2 m wide");
    }
    if (config.box_count < 0 || config.thin_box_count < 0 || config.thin_box_count > config.box_count)
    {
        throw std::invalid_argument("procedural box counts are invalid");
    }
    if (config.box_count > 100000)
    {
        throw std::invalid_argument("procedural box_count is unreasonably large");
    }

    ProceduralScene scene;
    scene.config = config;

    const double length = config.ground_length_m;
    const double width = config.ground_width_m;
    const double halfWidth = 0.5 * width;

    // Ground: x in [0, length], y in [-width/2, width/2], z = 0. Two triangles
    // wound so the +z side is outward.
    scene.vertices.emplace_back(0.0, -halfWidth, 0.0);
    scene.vertices.emplace_back(length, -halfWidth, 0.0);
    scene.vertices.emplace_back(length, halfWidth, 0.0);
    scene.vertices.emplace_back(0.0, halfWidth, 0.0);
    AppendFace(scene, 0U, 1U, 2U);
    AppendFace(scene, 0U, 2U, 3U);

    std::mt19937_64 rng(config.seed);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    const auto between = [&](const double low, const double high) { return low + (high - low) * unit(rng); };

    const double maxLengthX = std::min(10.0, length - 4.0);
    const double maxDepthY = std::min(6.0, halfWidth * 0.8);
    const int regularCount = config.box_count - config.thin_box_count;

    for (int index = 0; index < config.box_count; ++index)
    {
        const bool thin = index >= regularCount;
        double sizeX = thin ? between(0.5, 3.0) : between(3.0, maxLengthX);
        double sizeY = thin ? between(0.05, 0.15) : between(2.0, maxDepthY);
        const double sizeZ = thin ? between(2.5, 4.0) : between(1.5, 6.0);

        sizeX = std::min(sizeX, std::max(0.05, length - 4.0));
        sizeY = std::min(sizeY, std::max(0.05, halfWidth - 1.0));

        const double xLow = 2.0;
        const double xHigh = length - sizeX - 2.0;
        const double x0 = xHigh > xLow ? between(xLow, xHigh) : xLow;

        const int side = (index % 2 == 0) ? 1 : -1;
        double y0 = 0.0;
        if (side > 0)
        {
            const double low = 0.5;
            const double high = halfWidth - sizeY - 0.5;
            y0 = high > low ? between(low, high) : low;
        }
        else
        {
            const double low = -halfWidth + 0.5;
            const double high = -0.5 - sizeY;
            y0 = high > low ? between(low, high) : low;
        }

        const Eigen::Vector3d minimum(x0, y0, 0.0);
        const Eigen::Vector3d maximum(x0 + sizeX, y0 + sizeY, sizeZ);
        AppendBox(scene, minimum, maximum);
    }

    Eigen::Vector3d bboxMin = scene.vertices.front();
    Eigen::Vector3d bboxMax = scene.vertices.front();
    for (const auto &vertex : scene.vertices)
    {
        bboxMin = bboxMin.cwiseMin(vertex);
        bboxMax = bboxMax.cwiseMax(vertex);
    }
    scene.bbox_min = bboxMin;
    scene.bbox_max = bboxMax;
    return scene;
}

std::optional<AnalyticHit> RaycastScene(const ProceduralScene &scene, const Eigen::Vector3d &origin,
                                        const Eigen::Vector3d &dir, const double t_min, const double t_max)
{
    std::optional<AnalyticHit> best;

    const auto consider = [&best](const double t, const std::uint32_t primitiveId) {
        if (!best.has_value() || t < best->t || (t == best->t && primitiveId < best->primitive_id))
        {
            best = AnalyticHit{t, primitiveId};
        }
    };

    // Ground plane, primitive id 0.
    const double length = scene.config.ground_length_m;
    const double width = scene.config.ground_width_m;
    const auto ground =
        RaycastPlane(origin, dir, Eigen::Vector3d(0.5 * length, 0.0, 0.0), Eigen::Vector3d(1.0, 0.0, 0.0),
                     Eigen::Vector3d(0.0, 1.0, 0.0), 0.5 * length, 0.5 * width, t_min, t_max);
    if (ground.has_value())
    {
        consider(*ground, 0U);
    }

    // Axis-aligned boxes, primitive id = box index + 1. Closed-form placeholder,
    // not a BVH or a production ray caster.
    for (std::size_t index = 0; index < scene.boxes.size(); ++index)
    {
        const auto hit = RaycastAabb(origin, dir, scene.boxes[index].first, scene.boxes[index].second, t_min, t_max);
        if (hit.has_value())
        {
            consider(*hit, static_cast<std::uint32_t>(index + 1U));
        }
    }
    return best;
}

void WriteProceduralPly(const fs::path &path, const ProceduralScene &scene)
{
    std::vector<std::uint8_t> bytes;
    std::ostringstream header;
    header << "ply\n"
           << "format binary_little_endian 1.0\n"
           << "comment frame W\n"
           << "comment units m\n"
           << "element vertex " << scene.vertices.size() << "\n"
           << "property double x\n"
           << "property double y\n"
           << "property double z\n"
           << "element face " << scene.indices.size() << "\n"
           << "property list uchar int vertex_indices\n"
           << "end_header\n";
    const std::string headerText = header.str();
    bytes.reserve(headerText.size() + scene.vertices.size() * 24U + scene.indices.size() * 13U);
    bytes.insert(bytes.end(), headerText.begin(), headerText.end());
    for (const auto &vertex : scene.vertices)
    {
        PutF64(bytes, vertex.x());
        PutF64(bytes, vertex.y());
        PutF64(bytes, vertex.z());
    }
    for (const auto &face : scene.indices)
    {
        bytes.push_back(3U);
        PutI32(bytes, static_cast<std::int32_t>(face[0]));
        PutI32(bytes, static_cast<std::int32_t>(face[1]));
        PutI32(bytes, static_cast<std::int32_t>(face[2]));
    }

    fs::path temporary = path;
    temporary += ".tmp";
    if (path.has_parent_path())
    {
        raylidar::EnsureDir(path.parent_path());
    }
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output)
        {
            throw IoError("cannot write " + temporary.string());
        }
        output.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        if (!output)
        {
            throw IoError("failed to write " + temporary.string());
        }
    }
    std::error_code error;
    fs::rename(temporary, path, error);
    if (error)
    {
        std::error_code ignored;
        fs::remove(temporary, ignored);
        throw IoError("cannot rename " + temporary.string() + " to " + path.string());
    }
}

nlohmann::json BuildSceneManifest(const ProceduralScene &scene, const fs::path &plyPath)
{
    nlohmann::json manifest;
    manifest["schema_version"] = 1;
    manifest["frame"] = "W";
    manifest["units"] = "m";
    manifest["generator"] = "procedural_street_v1";
    manifest["seed"] = scene.config.seed;
    manifest["ground_length_m"] = scene.config.ground_length_m;
    manifest["ground_width_m"] = scene.config.ground_width_m;
    manifest["box_count"] = scene.config.box_count;
    manifest["thin_box_count"] = scene.config.thin_box_count;
    manifest["vertex_count"] = static_cast<std::uint64_t>(scene.vertices.size());
    manifest["triangle_count"] = static_cast<std::uint64_t>(scene.indices.size());
    manifest["bbox_min_m"] = nlohmann::json::array({scene.bbox_min.x(), scene.bbox_min.y(), scene.bbox_min.z()});
    manifest["bbox_max_m"] = nlohmann::json::array({scene.bbox_max.x(), scene.bbox_max.y(), scene.bbox_max.z()});
    manifest["sha256"] = raylidar::Sha256File(plyPath);
    return manifest;
}
} // namespace raylidar::eval
