#include <raylidar/overlay/project.h>

#include <raylidar/common/io.h>
#include <raylidar/overlay/distance_transform.h>

#include <stb_image.h>
#include <stb_image_write.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace raylidar::overlay
{
namespace
{
namespace fs = std::filesystem;

constexpr std::int64_t kMaxInputPixels = 50'000'000;
constexpr std::int64_t kMaxOutputPixels = 64'000'000;

std::string FrameName(int frame)
{
    std::ostringstream out;
    out << std::setw(10) << std::setfill('0') << frame;
    return out.str();
}

bool HasComponent(const fs::path &path, std::string_view name)
{
    for (const fs::path &part : path)
    {
        if (part.string() == name)
        {
            return true;
        }
    }
    return false;
}

struct ImageDeleter
{
    void operator()(unsigned char *data) const noexcept { stbi_image_free(data); }
};

std::string ResolvePath(const fs::path &path)
{
    std::error_code error;
    const fs::path resolved = fs::weakly_canonical(path, error);
    if (error)
    {
        return path.string();
    }
    return resolved.string();
}

std::uint8_t Luma(std::uint8_t red, std::uint8_t green, std::uint8_t blue)
{
    const double value = 0.299 * static_cast<double>(red) + 0.587 * static_cast<double>(green) +
                         0.114 * static_cast<double>(blue);
    const double clamped = std::clamp(value, 0.0, 255.0);
    return static_cast<std::uint8_t>(std::lround(clamped));
}

std::vector<std::uint8_t> ToGray(const std::vector<std::uint8_t> &rgb, int width, int height)
{
    const std::size_t pixels =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    std::vector<std::uint8_t> gray(pixels);
    for (std::size_t index = 0; index < pixels; ++index)
    {
        gray[index] = Luma(rgb[index * 3U + 0U], rgb[index * 3U + 1U], rgb[index * 3U + 2U]);
    }
    return gray;
}

std::vector<std::uint8_t> UpscaleNearest(const std::vector<std::uint8_t> &rgb, int width,
                                         int height, int scale)
{
    const int outWidth = width * scale;
    const int outHeight = height * scale;
    std::vector<std::uint8_t> out(static_cast<std::size_t>(outWidth) *
                                  static_cast<std::size_t>(outHeight) * 3U);
    for (int y = 0; y < outHeight; ++y)
    {
        const int sourceY = y / scale;
        for (int x = 0; x < outWidth; ++x)
        {
            const int sourceX = x / scale;
            const std::size_t source =
                (static_cast<std::size_t>(sourceY) * static_cast<std::size_t>(width) +
                 static_cast<std::size_t>(sourceX)) *
                3U;
            const std::size_t target =
                (static_cast<std::size_t>(y) * static_cast<std::size_t>(outWidth) +
                 static_cast<std::size_t>(x)) *
                3U;
            out[target + 0U] = rgb[source + 0U];
            out[target + 1U] = rgb[source + 1U];
            out[target + 2U] = rgb[source + 2U];
        }
    }
    return out;
}

std::vector<std::uint8_t> Crop(const std::vector<std::uint8_t> &rgb, int width, int x0, int y0,
                               int cropWidth, int cropHeight)
{
    std::vector<std::uint8_t> out(static_cast<std::size_t>(cropWidth) *
                                  static_cast<std::size_t>(cropHeight) * 3U);
    for (int y = 0; y < cropHeight; ++y)
    {
        const std::size_t source =
            (static_cast<std::size_t>(y0 + y) * static_cast<std::size_t>(width) +
             static_cast<std::size_t>(x0)) *
            3U;
        const std::size_t target = static_cast<std::size_t>(y) * static_cast<std::size_t>(cropWidth) *
                                   3U;
        std::memcpy(out.data() + target, rgb.data() + source,
                    static_cast<std::size_t>(cropWidth) * 3U);
    }
    return out;
}

void WritePng(const fs::path &path, int width, int height, const std::vector<std::uint8_t> &rgb)
{
    if (stbi_write_png(path.string().c_str(), width, height, 3, rgb.data(), width * 3) == 0)
    {
        throw IoError("cannot write overlay PNG " + path.string(), 4);
    }
}

struct CellHash
{
    std::size_t operator()(const std::pair<int, int> &cell) const noexcept
    {
        const std::uint64_t high = static_cast<std::uint32_t>(cell.first);
        const std::uint64_t low = static_cast<std::uint32_t>(cell.second);
        return static_cast<std::size_t>((high << 32U) ^ low);
    }
};
} // namespace

ProjectionResult ProjectPoints(const Eigen::MatrixXd &pointsV, const Eigen::Matrix4d &T_Ck_rect_V,
                               const Eigen::Matrix3d &K, int imageWidth, int imageHeight,
                               double minRangeM, double maxRangeM, int marginPx, bool clampRange)
{
    if (pointsV.cols() != 3 && pointsV.cols() != 4)
    {
        throw std::invalid_argument("points must be an (N,3) or (N,4) matrix");
    }
    if (imageWidth <= 0 || imageHeight <= 0)
    {
        throw std::invalid_argument("image dimensions must be positive");
    }

    ProjectionResult result;
    const Eigen::Index count = pointsV.rows();
    const std::size_t size = static_cast<std::size_t>(count);
    result.imageWidth = imageWidth;
    result.imageHeight = imageHeight;
    result.uv.resize(size);
    result.z.resize(size);
    result.rangeM.resize(size);
    result.valid.resize(size);
    result.sourceIndex.resize(size);

    const double fx = K(0, 0);
    const double cx = K(0, 2);
    const double fy = K(1, 1);
    const double cy = K(1, 2);
    const double margin = static_cast<double>(std::max(marginPx, 0));
    const double maxU = static_cast<double>(imageWidth - 1) - margin;
    const double maxV = static_cast<double>(imageHeight - 1) - margin;

    for (Eigen::Index row = 0; row < count; ++row)
    {
        const std::size_t index = static_cast<std::size_t>(row);
        Eigen::Vector4d homogeneous;
        homogeneous << pointsV(row, 0), pointsV(row, 1), pointsV(row, 2), 1.0;
        const Eigen::Vector3d pointC = (T_Ck_rect_V * homogeneous).head<3>();
        const double z = pointC.z();
        const double range = pointC.norm();

        result.sourceIndex[index] = index;
        result.z[index] = z;
        result.rangeM[index] = range;

        const bool finite = pointC.allFinite() && std::isfinite(z) && std::isfinite(range);
        if (!(finite && z > 0.0))
        {
            result.uv[index] = Eigen::Vector2d(std::numeric_limits<double>::quiet_NaN(),
                                               std::numeric_limits<double>::quiet_NaN());
            result.valid[index] = false;
            continue;
        }

        const double u = fx * pointC.x() / z + cx;
        const double v = fy * pointC.y() / z + cy;
        result.uv[index] = Eigen::Vector2d(u, v);

        const bool inRange = !clampRange || (range >= minRangeM && range <= maxRangeM);
        const bool inBounds = std::isfinite(u) && std::isfinite(v) && u >= margin && u <= maxU &&
                              v >= margin && v <= maxV;
        result.valid[index] = finite && z > 0.0 && inRange && inBounds;
    }
    return result;
}

std::vector<std::uint8_t> RenderOverlay(const std::vector<std::uint8_t> &rgb, int width,
                                        int height, const ProjectionResult &result, int scale,
                                        int radius, int gridPx)
{
    if (width <= 0 || height <= 0)
    {
        throw std::invalid_argument("image dimensions must be positive");
    }
    const std::size_t pixels =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    if (rgb.size() != pixels * 3U)
    {
        throw std::invalid_argument("RGB buffer size does not match width * height * 3");
    }
    if (scale < 1)
    {
        throw std::invalid_argument("scale must be >= 1");
    }

    std::vector<std::uint8_t> canvas = rgb;
    if (gridPx > 0)
    {
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                if (x % gridPx == 0 || y % gridPx == 0)
                {
                    const std::size_t index =
                        (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                         static_cast<std::size_t>(x)) *
                        3U;
                    canvas[index + 0U] = 200U;
                    canvas[index + 1U] = 200U;
                    canvas[index + 2U] = 200U;
                }
            }
        }
    }

    const int half = std::max(radius, 0);
    for (std::size_t index = 0; index < result.uv.size(); ++index)
    {
        if (index >= result.valid.size() || !result.valid[index])
        {
            continue;
        }
        const Eigen::Vector2d &point = result.uv[index];
        if (!point.allFinite())
        {
            continue;
        }
        const int u = static_cast<int>(std::lround(point.x()));
        const int v = static_cast<int>(std::lround(point.y()));
        const double range = std::clamp(result.rangeM[index], 10.0, 30.0);
        const double t = (range - 10.0) / 20.0;
        const std::uint8_t red = static_cast<std::uint8_t>(std::lround(255.0 * t));
        const std::uint8_t green = static_cast<std::uint8_t>(std::lround(255.0 * (1.0 - t)));
        const std::uint8_t blue = 255U;

        for (int dy = -half; dy <= half; ++dy)
        {
            const int y = v + dy;
            if (y < 0 || y >= height)
            {
                continue;
            }
            for (int dx = -half; dx <= half; ++dx)
            {
                const int x = u + dx;
                if (x < 0 || x >= width)
                {
                    continue;
                }
                const std::size_t pixel =
                    (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                     static_cast<std::size_t>(x)) *
                    3U;
                canvas[pixel + 0U] = red;
                canvas[pixel + 1U] = green;
                canvas[pixel + 2U] = blue;
            }
        }
    }

    if (scale == 1)
    {
        return canvas;
    }
    return UpscaleNearest(canvas, width, height, scale);
}

nlohmann::json EdgeDistanceScore(const std::vector<std::uint8_t> &gray, int width, int height,
                                 const ProjectionResult &result, double discontinuityM,
                                 double neighbourPx)
{
    nlohmann::json score;
    score["median_px"] = std::numeric_limits<double>::quiet_NaN();
    score["p90_px"] = std::numeric_limits<double>::quiet_NaN();
    score["count"] = 0;
    score["fraction_within_2px"] = 0.0;
    score["fallback_all_points"] = false;

    std::vector<std::size_t> valid;
    valid.reserve(result.valid.size());
    for (std::size_t index = 0; index < result.valid.size(); ++index)
    {
        if (result.valid[index])
        {
            valid.push_back(index);
        }
    }
    if (valid.empty())
    {
        return score;
    }

    const std::vector<double> magnitude = GradientMagnitude(gray, width, height);
    const double threshold = Percentile(magnitude, 0.90);
    std::vector<std::uint8_t> edges(magnitude.size(), 0U);
    for (std::size_t index = 0; index < magnitude.size(); ++index)
    {
        edges[index] = magnitude[index] > threshold ? 1U : 0U;
    }
    const std::vector<double> distance = EuclideanDistanceTransform(edges, width, height);

    // Depth-discontinuity candidates via a uniform image-space grid.
    std::vector<std::size_t> candidates;
    const int cellSize = std::max(1, static_cast<int>(std::lround(neighbourPx)));
    if (discontinuityM >= 0.0 && neighbourPx > 0.0)
    {
        std::unordered_map<std::pair<int, int>, std::vector<std::size_t>, CellHash> grid;
        grid.reserve(valid.size());
        for (const std::size_t index : valid)
        {
            const int cx = static_cast<int>(
                std::floor(result.uv[index].x() / static_cast<double>(cellSize)));
            const int cy = static_cast<int>(
                std::floor(result.uv[index].y() / static_cast<double>(cellSize)));
            grid[{cx, cy}].push_back(index);
        }

        for (const std::size_t index : valid)
        {
            const double u = result.uv[index].x();
            const double v = result.uv[index].y();
            const int cx = static_cast<int>(std::floor(u / static_cast<double>(cellSize)));
            const int cy = static_cast<int>(std::floor(v / static_cast<double>(cellSize)));
            bool candidate = false;
            for (int oy = -1; oy <= 1 && !candidate; ++oy)
            {
                for (int ox = -1; ox <= 1 && !candidate; ++ox)
                {
                    const auto found = grid.find({cx + ox, cy + oy});
                    if (found == grid.end())
                    {
                        continue;
                    }
                    for (const std::size_t other : found->second)
                    {
                        if (other == index || !result.valid[other])
                        {
                            continue;
                        }
                        const double dx = result.uv[other].x() - u;
                        const double dy = result.uv[other].y() - v;
                        if (std::hypot(dx, dy) > neighbourPx)
                        {
                            continue;
                        }
                        if (std::abs(result.z[index] - result.z[other]) > discontinuityM)
                        {
                            candidate = true;
                            break;
                        }
                    }
                }
            }
            if (candidate)
            {
                candidates.push_back(index);
            }
        }
    }

    bool fallback = false;
    std::vector<std::size_t> chosen;
    if (candidates.size() < 50U)
    {
        fallback = true;
        chosen = valid;
    }
    else
    {
        chosen = std::move(candidates);
    }

    std::vector<double> finite;
    finite.reserve(chosen.size());
    std::size_t sampled = 0;
    for (const std::size_t index : chosen)
    {
        const Eigen::Vector2d &point = result.uv[index];
        if (!point.allFinite())
        {
            continue;
        }
        const int u = static_cast<int>(std::lround(point.x()));
        const int v = static_cast<int>(std::lround(point.y()));
        if (u < 0 || u >= width || v < 0 || v >= height)
        {
            continue;
        }
        ++sampled;
        const double value =
            distance[static_cast<std::size_t>(v) * static_cast<std::size_t>(width) +
                     static_cast<std::size_t>(u)];
        if (std::isfinite(value))
        {
            finite.push_back(value);
        }
    }

    score["count"] = sampled;
    score["fallback_all_points"] = fallback;
    if (!finite.empty())
    {
        score["median_px"] = Percentile(finite, 0.5);
        score["p90_px"] = Percentile(finite, 0.9);
        const std::size_t within = static_cast<std::size_t>(
            std::count_if(finite.begin(), finite.end(), [](double value) { return value <= 2.0; }));
        score["fraction_within_2px"] = static_cast<double>(within) /
                                       static_cast<double>(finite.size());
    }
    return score;
}

nlohmann::json OverlayFrame(const fs::path &imagePath, const Eigen::MatrixXd &pointsV,
                            const Eigen::Matrix4d &T_Ck_rect_V, const Eigen::Matrix3d &K,
                            const fs::path &outDir, int frame, std::string_view cam, int scale,
                            double minRangeM, double maxRangeM, int marginPx)
{
    if (cam.empty())
    {
        throw std::invalid_argument("camera name must not be empty");
    }

    const std::string encoded = imagePath.string();
    int width = 0;
    int height = 0;
    int channels = 0;
    if (stbi_info(encoded.c_str(), &width, &height, &channels) == 0)
    {
        throw IoError("cannot read image header: " + encoded, 4);
    }
    if (width <= 0 || height <= 0)
    {
        throw IoError("image has invalid dimensions: " + encoded, 4);
    }
    const std::int64_t pixelCount = static_cast<std::int64_t>(width) *
                                    static_cast<std::int64_t>(height);
    if (pixelCount > kMaxInputPixels)
    {
        throw std::invalid_argument("image too large");
    }

    std::unique_ptr<unsigned char, ImageDeleter> decoded(
        stbi_load(encoded.c_str(), &width, &height, &channels, 3));
    if (!decoded)
    {
        throw IoError("cannot decode image: " + encoded, 4);
    }
    const std::size_t pixels =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    std::vector<std::uint8_t> rgb(pixels * 3U);
    std::memcpy(rgb.data(), decoded.get(), pixels * 3U);

    const ProjectionResult projection = ProjectPoints(pointsV, T_Ck_rect_V, K, width, height,
                                                      minRangeM, maxRangeM, marginPx, true);
    const std::vector<std::uint8_t> overlay1x = RenderOverlay(rgb, width, height, projection, 1, 1, 80);

    bool any = false;
    int minU = width - 1;
    int minV = height - 1;
    int maxU = 0;
    int maxV = 0;
    std::size_t validCount = 0;
    for (std::size_t index = 0; index < projection.valid.size(); ++index)
    {
        if (!projection.valid[index])
        {
            continue;
        }
        ++validCount;
        const int u = static_cast<int>(std::floor(projection.uv[index].x()));
        const int v = static_cast<int>(std::floor(projection.uv[index].y()));
        if (!any)
        {
            minU = maxU = u;
            minV = maxV = v;
            any = true;
        }
        else
        {
            minU = std::min(minU, u);
            minV = std::min(minV, v);
            maxU = std::max(maxU, u);
            maxV = std::max(maxV, v);
        }
    }

    int cropX = 0;
    int cropY = 0;
    int cropWidth = width;
    int cropHeight = height;
    if (any)
    {
        cropX = std::clamp(minU - 40, 0, width - 1);
        cropY = std::clamp(minV - 40, 0, height - 1);
        const int right = std::clamp(maxU + 40, 0, width - 1);
        const int bottom = std::clamp(maxV + 40, 0, height - 1);
        cropWidth = right - cropX + 1;
        cropHeight = bottom - cropY + 1;
    }

    int effectiveScale = std::max(scale, 1);
    while (effectiveScale > 1 &&
           static_cast<std::int64_t>(cropWidth) * static_cast<std::int64_t>(cropHeight) *
                   static_cast<std::int64_t>(effectiveScale) * static_cast<std::int64_t>(effectiveScale) >
               kMaxOutputPixels)
    {
        --effectiveScale;
    }

    const std::vector<std::uint8_t> cropped =
        Crop(overlay1x, width, cropX, cropY, cropWidth, cropHeight);
    const std::vector<std::uint8_t> overlay4x =
        effectiveScale == 1 ? cropped
                            : UpscaleNearest(cropped, cropWidth, cropHeight, effectiveScale);

    const fs::path frameDir = outDir / ("frame_" + FrameName(frame));
    EnsureDir(frameDir);
    const fs::path path1x = frameDir / (std::string(cam) + "_overlay_1x.png");
    const fs::path path4x = frameDir / (std::string(cam) + "_overlay_4x.png");
    WritePng(path1x, width, height, overlay1x);
    WritePng(path4x, cropWidth * effectiveScale, cropHeight * effectiveScale, overlay4x);

    const std::vector<std::uint8_t> gray = ToGray(overlay1x, width, height);
    const nlohmann::json score = EdgeDistanceScore(gray, width, height, projection);

    bool invariant = true;
    for (std::size_t index = 0; index < projection.valid.size(); ++index)
    {
        if (!projection.valid[index])
        {
            continue;
        }
        const long u = std::lround(projection.uv[index].x());
        const long v = std::lround(projection.uv[index].y());
        if (u < 0 || u >= width || v < 0 || v >= height)
        {
            invariant = false;
            break;
        }
    }

    nlohmann::json out;
    out["image"] = encoded;
    out["image_resolved"] = ResolvePath(imagePath);
    out["rectified"] = HasComponent(imagePath, "data_rect");
    out["image_size"] = nlohmann::json::array({width, height});
    out["scale_factor"] = effectiveScale;
    out["points_total"] = static_cast<std::size_t>(pointsV.rows());
    out["points_valid"] = validCount;
    if (any)
    {
        out["projected_bbox"] = nlohmann::json::array({minU, minV, maxU, maxV});
    }
    else
    {
        out["projected_bbox"] = nlohmann::json::array();
    }
    out["overlay_1x"] = path1x.string();
    out["overlay_4x"] = path4x.string();
    out["projection_invariant"] = invariant;
    for (const char *key : {"median_px", "p90_px", "count", "fraction_within_2px",
                            "fallback_all_points"})
    {
        out[key] = score[key];
    }
    return out;
}
} // namespace raylidar::overlay
