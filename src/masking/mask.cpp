#include <raylidar/masking/mask.h>

#include <raylidar/common/io.h>
#include <raylidar/overlay/project.h>

#include <stb_image_write.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace raylidar::masking
{
namespace
{
namespace fs = std::filesystem;

constexpr std::uint8_t kBitDynamic2D = static_cast<std::uint8_t>(kDynamic2D);
constexpr std::uint8_t kBitDynamicBBox = static_cast<std::uint8_t>(kDynamicBBox);
constexpr std::uint8_t kBitDynamicUnion = static_cast<std::uint8_t>(kDynamicUnion);
constexpr std::uint8_t kBitEgoRoi = static_cast<std::uint8_t>(kEgoRoi);
constexpr std::uint8_t kBitKeptStatic = static_cast<std::uint8_t>(kKeptStatic);
constexpr std::uint8_t kBitInvalid = static_cast<std::uint8_t>(kInvalidNonFinite);
[[nodiscard]] std::string FrameName(const int frame)
{
    char buffer[16] = {};
    std::snprintf(buffer, sizeof(buffer), "%010d", frame);
    return std::string(buffer);
}

[[nodiscard]] bool InEgoRoi(const Eigen::Vector3d &point, const MaskOptions &options)
{
    return point.x() > options.egoRoiXMin && point.x() < options.egoRoiXMax &&
           point.y() > options.egoRoiYMin && point.y() < options.egoRoiYMax &&
           point.z() > options.egoRoiZMin && point.z() < options.egoRoiZMax;
}

struct CameraProjection
{
    std::string name;
    Eigen::Matrix3d intrinsics = Eigen::Matrix3d::Identity();
    Eigen::Matrix4d T_Ck_rect_V = Eigen::Matrix4d::Identity();
    const SemanticImage *semantics = nullptr;
    int width = 0;
    int height = 0;
    overlay::ProjectionResult projection;
};

struct CellKey
{
    long long x = 0;
    long long y = 0;
    long long z = 0;

    bool operator==(const CellKey &other) const noexcept
    {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct CellHash
{
    std::size_t operator()(const CellKey &key) const noexcept
    {
        const auto mix = [](std::uint64_t value) {
            value ^= value >> 33U;
            value *= 0xff51afd7ed558ccdULL;
            value ^= value >> 33U;
            return value;
        };
        const std::uint64_t packed =
            (static_cast<std::uint64_t>(static_cast<std::uint32_t>(key.x)) << 42U) ^
            (static_cast<std::uint64_t>(static_cast<std::uint32_t>(key.y)) << 21U) ^
            static_cast<std::uint32_t>(key.z);
        return static_cast<std::size_t>(mix(packed));
    }
};

[[nodiscard]] CellKey CellOf(const Eigen::Vector3d &point, const double cell)
{
    return CellKey{static_cast<long long>(std::floor(point.x() / cell)),
                   static_cast<long long>(std::floor(point.y() / cell)),
                   static_cast<long long>(std::floor(point.z() / cell))};
}

void CopyDynamicSemanticPixels(const CameraProjection &camera, std::vector<std::uint8_t> *pixels)
{
    const SemanticImage &semantics = *camera.semantics;
    const std::size_t count =
        static_cast<std::size_t>(camera.width) * static_cast<std::size_t>(camera.height);
    pixels->assign(count, 0U);
    if (semantics.labels.size() < count)
    {
        throw IoError("semantic image is smaller than its declared dimensions",
                      4);
    }
    for (std::size_t index = 0; index < count; ++index)
    {
        if (IsDynamicLabel(semantics.labels[index]))
        {
            (*pixels)[index] = 1U;
        }
    }
}

void RasterizeBox(const DynamicBBox &box, const CameraProjection &camera,
                  std::vector<std::uint8_t> *pixels)
{
    double minU = std::numeric_limits<double>::infinity();
    double maxU = -std::numeric_limits<double>::infinity();
    double minV = std::numeric_limits<double>::infinity();
    double maxV = -std::numeric_limits<double>::infinity();
    const double fx = camera.intrinsics(0, 0);
    const double cx = camera.intrinsics(0, 2);
    const double fy = camera.intrinsics(1, 1);
    const double cy = camera.intrinsics(1, 2);
    bool any = false;

    for (int xSign = -1; xSign <= 1; xSign += 2)
    {
        for (int ySign = -1; ySign <= 1; ySign += 2)
        {
            for (int zSign = -1; zSign <= 1; zSign += 2)
            {
                const Eigen::Vector4d corner(0.5 * static_cast<double>(xSign),
                                             0.5 * static_cast<double>(ySign),
                                             0.5 * static_cast<double>(zSign), 1.0);
                const Eigen::Vector3d world = (box.T_W_B * corner).head<3>();
                const Eigen::Vector3d cameraPoint =
                    (camera.T_Ck_rect_V *
                     Eigen::Vector4d(world.x(), world.y(), world.z(), 1.0))
                        .head<3>();
                if (!cameraPoint.allFinite() || !(cameraPoint.z() > 0.0))
                {
                    continue;
                }
                const double u = fx * cameraPoint.x() / cameraPoint.z() + cx;
                const double v = fy * cameraPoint.y() / cameraPoint.z() + cy;
                if (!std::isfinite(u) || !std::isfinite(v))
                {
                    continue;
                }
                minU = std::min(minU, u);
                maxU = std::max(maxU, u);
                minV = std::min(minV, v);
                maxV = std::max(maxV, v);
                any = true;
            }
        }
    }
    if (!any)
    {
        return;
    }

    const int x0 = std::clamp(static_cast<int>(std::floor(minU)), 0, camera.width - 1);
    const int x1 = std::clamp(static_cast<int>(std::ceil(maxU)), 0, camera.width - 1);
    const int y0 = std::clamp(static_cast<int>(std::floor(minV)), 0, camera.height - 1);
    const int y1 = std::clamp(static_cast<int>(std::ceil(maxV)), 0, camera.height - 1);
    if (x0 > x1 || y0 > y1)
    {
        return;
    }
    for (int row = y0; row <= y1; ++row)
    {
        const std::size_t base =
            static_cast<std::size_t>(row) * static_cast<std::size_t>(camera.width);
        for (int column = x0; column <= x1; ++column)
        {
            (*pixels)[base + static_cast<std::size_t>(column)] = 1U;
        }
    }
}

void WriteBinaryPng(const fs::path &path, const std::vector<std::uint8_t> &pixels,
                    const int width, const int height)
{
    if (width <= 0 || height <= 0 ||
        pixels.size() != static_cast<std::size_t>(width) * static_cast<std::size_t>(height))
    {
        throw std::invalid_argument("pixel mask does not match its image dimensions");
    }
    std::vector<std::uint8_t> bytes(pixels.size());
    for (std::size_t index = 0; index < pixels.size(); ++index)
    {
        bytes[index] = pixels[index] == 0U ? 0U : 255U;
    }
    std::error_code error;
    fs::create_directories(path.parent_path(), error);
    if (stbi_write_png(path.string().c_str(), width, height, 1, bytes.data(), width) == 0)
    {
        throw IoError("cannot write mask PNG " + path.string(), 4);
    }
}

void ValidateOptions(const MaskOptions &options)
{
    if (!(options.egoRoiXMin < options.egoRoiXMax) ||
        !(options.egoRoiYMin < options.egoRoiYMax) ||
        !(options.egoRoiZMin < options.egoRoiZMax) ||
        options.bboxMaxTimeGapFrames < 0 || !(options.residualDynamicRadiusM > 0.0))
    {
        throw std::invalid_argument("invalid mask option bounds");
    }
}
} // namespace

MaskResult BuildMask(const std::vector<lidar::PointXYZI> &rawPoints,
                     const std::vector<Eigen::Vector3d> &uncurledPoints, const int frame,
                     const calib::Kitti360Calib &calib, const Eigen::Matrix4d &T_W_V,
                     const std::vector<DynamicBBox> &boxes,
                     const std::map<std::string, SemanticImage> &semantics,
                     const std::map<std::string, std::pair<int, int>> &imageSizes,
                     const std::map<std::string, Eigen::Matrix4d> &T_Ck_rect_V,
                     const MaskOptions &options)
{
    (void)frame;
    if (rawPoints.size() != uncurledPoints.size())
    {
        throw std::invalid_argument("raw and uncurled point vectors must be index-aligned");
    }
    ValidateOptions(options);

    const std::size_t count = rawPoints.size();
    MaskResult result;
    result.bits.assign(count, 0U);

    std::vector<CameraProjection> cameras;
    Eigen::MatrixXd uncurledMatrix(static_cast<Eigen::Index>(count), 3);
    for (std::size_t index = 0; index < count; ++index)
    {
        uncurledMatrix.row(static_cast<Eigen::Index>(index)) = uncurledPoints[index].transpose();
    }

    for (const auto &[name, transform] : T_Ck_rect_V)
    {
        const auto semantic = semantics.find(name);
        const auto size = imageSizes.find(name);
        const auto intrinsics = calib.K_rect.find(name);
        if (semantic == semantics.end() || size == imageSizes.end() ||
            intrinsics == calib.K_rect.end())
        {
            continue;
        }
        if (size->second.first <= 0 || size->second.second <= 0)
        {
            throw std::invalid_argument("image dimensions must be positive");
        }
        CameraProjection camera;
        camera.name = name;
        camera.intrinsics = intrinsics->second;
        camera.T_Ck_rect_V = transform;
        camera.semantics = &semantic->second;
        camera.width = size->second.first;
        camera.height = size->second.second;
        camera.projection =
            overlay::ProjectPoints(uncurledMatrix, transform, camera.intrinsics, camera.width,
                                   camera.height, 0.0, std::numeric_limits<double>::infinity(),
                                   0, false);
        cameras.push_back(std::move(camera));
    }

    const std::vector<const DynamicBBox *> applicable =
        ApplicableBoxes(boxes, frame, options.bboxMaxTimeGapFrames);

    std::size_t nonfinite = 0;
    std::size_t egoRoi = 0;
    std::size_t dynamic2d = 0;
    std::size_t dynamicBbox = 0;
    std::size_t dynamicUnion = 0;
    std::size_t keptStatic = 0;
    std::size_t overlap2dOnly = 0;
    std::size_t overlapBboxOnly = 0;
    std::size_t overlapBoth = 0;

    std::vector<Eigen::Vector3d> worldPoints(count, Eigen::Vector3d::Zero());
    std::vector<bool> isKeptStatic(count, false);

    for (std::size_t index = 0; index < count; ++index)
    {
        const lidar::PointXYZI &raw = rawPoints[index];
        const Eigen::Vector3d rawPoint(raw.x, raw.y, raw.z);
        const Eigen::Vector3d uncurled = uncurledPoints[index];
        if (!rawPoint.allFinite() || !uncurled.allFinite())
        {
            result.bits[index] = kBitInvalid;
            ++nonfinite;
            continue;
        }

        std::uint8_t bits = 0U;
        const bool ego = InEgoRoi(rawPoint, options);
        if (ego)
        {
            bits |= kBitEgoRoi;
            ++egoRoi;
        }

        bool isDynamic2d = false;
        for (const CameraProjection &camera : cameras)
        {
            const double z = camera.projection.z[index];
            const Eigen::Vector2d uv = camera.projection.uv[index];
            if (!(z > 0.0) || !uv.allFinite() || uv.x() < 0.0 || uv.x() >= camera.width ||
                uv.y() < 0.0 || uv.y() >= camera.height)
            {
                continue;
            }
            if (IsDynamicLabel(LabelAt(*camera.semantics, uv.x(), uv.y())))
            {
                isDynamic2d = true;
                break;
            }
        }

        const Eigen::Vector3d world =
            (T_W_V * Eigen::Vector4d(uncurled.x(), uncurled.y(), uncurled.z(), 1.0)).head<3>();
        worldPoints[index] = world;
        bool isDynamicBbox = false;
        for (const DynamicBBox *box : applicable)
        {
            if (PointInBBox(world, *box))
            {
                isDynamicBbox = true;
                break;
            }
        }

        if (isDynamic2d)
        {
            bits |= kBitDynamic2D;
            ++dynamic2d;
        }
        if (isDynamicBbox)
        {
            bits |= kBitDynamicBBox;
            ++dynamicBbox;
        }
        if (isDynamic2d && isDynamicBbox)
        {
            bits |= kBitDynamicUnion;
            ++dynamicUnion;
            ++overlapBoth;
        }
        else if (isDynamic2d)
        {
            bits |= kBitDynamicUnion;
            ++dynamicUnion;
            ++overlap2dOnly;
        }
        else if (isDynamicBbox)
        {
            bits |= kBitDynamicUnion;
            ++dynamicUnion;
            ++overlapBboxOnly;
        }

        if (!ego && !isDynamic2d && !isDynamicBbox)
        {
            bits |= kBitKeptStatic;
            ++keptStatic;
            isKeptStatic[index] = true;
        }
        result.bits[index] = bits;
    }

    for (CameraProjection &camera : cameras)
    {
        std::vector<std::uint8_t> pixels;
        CopyDynamicSemanticPixels(camera, &pixels);
        for (const DynamicBBox *box : applicable)
        {
            RasterizeBox(*box, camera, &pixels);
        }
        if (camera.name == "image_00")
        {
            result.image_00_pixels = std::move(pixels);
        }
        else if (camera.name == "image_01")
        {
            result.image_01_pixels = std::move(pixels);
        }
        result.imageSizes[camera.name] = {camera.width, camera.height};
    }

    std::size_t residualInsideBbox = 0;
    std::size_t residualWithinRadius = 0;
    std::size_t residualEither = 0;

    const double radius = options.residualDynamicRadiusM;
    std::unordered_map<CellKey, std::vector<std::size_t>, CellHash> dynamicCells;
    dynamicCells.reserve(dynamicUnion * 2U + 1U);
    for (std::size_t index = 0; index < count; ++index)
    {
        if ((result.bits[index] & kBitDynamicUnion) != 0U && worldPoints[index].allFinite())
        {
            dynamicCells[CellOf(worldPoints[index], radius)].push_back(index);
        }
    }

    for (std::size_t index = 0; index < count; ++index)
    {
        if (!isKeptStatic[index])
        {
            continue;
        }
        const Eigen::Vector3d &world = worldPoints[index];
        bool inside = false;
        for (const DynamicBBox *box : applicable)
        {
            if (PointInBBox(world, *box))
            {
                inside = true;
                break;
            }
        }
        bool near = false;
        const CellKey center = CellOf(world, radius);
        const double radiusSquared = radius * radius;
        for (long long dx = -1; dx <= 1 && !near; ++dx)
        {
            for (long long dy = -1; dy <= 1 && !near; ++dy)
            {
                for (long long dz = -1; dz <= 1 && !near; ++dz)
                {
                    const auto bucket = dynamicCells.find(
                        CellKey{center.x + dx, center.y + dy, center.z + dz});
                    if (bucket == dynamicCells.end())
                    {
                        continue;
                    }
                    for (const std::size_t other : bucket->second)
                    {
                        if ((worldPoints[other] - world).squaredNorm() <= radiusSquared)
                        {
                            near = true;
                            break;
                        }
                    }
                }
            }
        }
        if (inside)
        {
            ++residualInsideBbox;
        }
        if (near)
        {
            ++residualWithinRadius;
        }
        if (inside || near)
        {
            ++residualEither;
        }
    }

    result.counts = {{"nonfinite", nonfinite},
                     {"ego_roi", egoRoi},
                     {"dynamic_2d", dynamic2d},
                     {"dynamic_bbox", dynamicBbox},
                     {"dynamic_union", dynamicUnion},
                     {"kept_static", keptStatic},
                     {"overlap_2d_only", overlap2dOnly},
                     {"overlap_bbox_only", overlapBboxOnly},
                     {"overlap_both", overlapBoth}};
    result.residual = {{"kept_static", keptStatic},
                       {"inside_bbox", residualInsideBbox},
                       {"within_radius", residualWithinRadius},
                       {"either", residualEither},
                       {"fraction", keptStatic == 0U
                                        ? 0.0
                                        : static_cast<double>(residualEither) /
                                              static_cast<double>(keptStatic)},
                       {"radius_m", radius}};
    return result;
}

nlohmann::json WriteMaskArtifact(const fs::path &masksDir, const int frame,
                                 const MaskResult &result,
                                 const std::vector<lidar::PointXYZI> &rawPoints,
                                 const std::string_view sequence)
{
    if (result.bits.size() != rawPoints.size())
    {
        throw std::invalid_argument("mask bit count does not match the raw point count");
    }
    const fs::path frameDir = masksDir / ("frame_" + FrameName(frame));
    raylidar::EnsureDir(frameDir);

    const fs::path pointsPath = frameDir / "points.bin";
    {
        std::ofstream output(pointsPath, std::ios::binary | std::ios::trunc);
        if (!output.good())
        {
            throw IoError("cannot write " + pointsPath.string(), 4);
        }
        output.write(reinterpret_cast<const char *>(result.bits.data()),
                     static_cast<std::streamsize>(result.bits.size()));
        if (!output.good())
        {
            throw IoError("error while writing " + pointsPath.string(), 4);
        }
    }

    const fs::path image00Path = frameDir / "image_00_dynamic.png";
    const fs::path image01Path = frameDir / "image_01_dynamic.png";
    const auto size00 = result.imageSizes.find("image_00");
    const auto size01 = result.imageSizes.find("image_01");
    if (size00 == result.imageSizes.end() || size01 == result.imageSizes.end())
    {
        throw std::invalid_argument("mask result is missing per-camera image dimensions");
    }
    WriteBinaryPng(image00Path, result.image_00_pixels, size00->second.first,
                   size00->second.second);
    WriteBinaryPng(image01Path, result.image_01_pixels, size01->second.first,
                   size01->second.second);

    nlohmann::json document;
    document["schema_version"] = 1;
    document["frame"] = frame;
    document["sequence"] = sequence;
    document["generated_at_utc"] = raylidar::UtcNowIso8601();
    document["point_count"] = rawPoints.size();
    document["counts"] = result.counts;
    document["residual_dynamic"] = result.residual;
    document["applies_to"] = {"images", "gt", "projections", "evaluation"};
    document["files"] = {{"points_sha256", raylidar::Sha256File(pointsPath)},
                         {"image_00_dynamic_sha256", raylidar::Sha256File(image00Path)},
                         {"image_01_dynamic_sha256", raylidar::Sha256File(image01Path)}};
    raylidar::WriteJson(frameDir / "mask.json", document);
    return document;
}

nlohmann::json BuildMaskManifest(const fs::path &masksDir,
                                 const std::vector<nlohmann::json> &frames,
                                 const std::string_view sequence,
                                 const std::string_view configSha256,
                                 const std::string_view manifestFileName)
{
    const std::array<const char *, 9> keys = {"nonfinite",       "ego_roi",     "dynamic_2d",
                                              "dynamic_bbox",    "dynamic_union",
                                              "kept_static",     "overlap_2d_only",
                                              "overlap_bbox_only", "overlap_both"};
    nlohmann::json totals = nlohmann::json::object();
    for (const char *key : keys)
    {
        totals[key] = std::size_t{0};
    }
    for (const nlohmann::json &frame : frames)
    {
        const nlohmann::json &counts = frame.at("counts");
        for (const char *key : keys)
        {
            totals[key] = totals[key].get<std::size_t>() + counts.at(key).get<std::size_t>();
        }
    }

    nlohmann::json document;
    document["schema_version"] = 1;
    document["sequence"] = sequence;
    document["generated_at_utc"] = raylidar::UtcNowIso8601();
    document["applies_to"] = {"images", "gt", "projections", "evaluation"};
    document["frames"] = frames;
    document["totals"] = totals;
    document["config_sha256"] = configSha256;

    raylidar::EnsureDir(masksDir);
    raylidar::WriteJson(masksDir / std::string(manifestFileName), document);
    return document;
}
} // namespace raylidar::masking
