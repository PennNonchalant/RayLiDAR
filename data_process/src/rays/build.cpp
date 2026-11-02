#include <raylidar/rays/build.h>

#include <raylidar/gt/cache.h>
#include <raylidar/rays/rays.h>

#include <Eigen/Core>

#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace raylidar::rays
{
std::vector<Ray> BuildRays(const std::vector<gt::GtPoint> &gtPoints, const int frame,
                           const std::map<std::string, Eigen::Matrix4d> &T_Ck_rect_V)
{
    if (frame < 0)
    {
        throw std::invalid_argument("ray frame id must be non-negative");
    }
    const auto cameraName = [](const std::uint8_t camera) -> const char * {
        switch (camera)
        {
        case 0U:
            return "image_00";
        case 1U:
            return "image_01";
        default:
            return nullptr;
        }
    };

    std::vector<Ray> rays;
    rays.reserve(gtPoints.size());
    for (const gt::GtPoint &point : gtPoints)
    {
        if (point.valid != 1U || point.camera == 255U)
        {
            continue;
        }
        const char *name = cameraName(point.camera);
        if (name == nullptr)
        {
            throw std::invalid_argument("GT point has an unknown camera id " + std::to_string(point.camera));
        }
        const auto transform = T_Ck_rect_V.find(name);
        if (transform == T_Ck_rect_V.end())
        {
            throw std::invalid_argument(std::string("missing T_Ck_rect_V for ") + name);
        }

        const Eigen::Vector4d homogeneous(static_cast<double>(point.p_L.x()), static_cast<double>(point.p_L.y()),
                                          static_cast<double>(point.p_L.z()), 1.0);
        const Eigen::Vector3d pointC = (transform->second * homogeneous).head<3>();
        const double range = pointC.norm();
        if (!pointC.allFinite() || !std::isfinite(range) || range <= 0.0)
        {
            throw std::invalid_argument("GT point projects to a non-finite camera position");
        }
        if (range < kRayMinRangeM || range > kRayMaxRangeM)
        {
            throw std::invalid_argument("GT point camera-frame range is outside [1.5, 60] m");
        }
        if (point.r_gt < 0.0F || std::abs(range - static_cast<double>(point.r_gt)) > 1.0e-3)
        {
            throw std::invalid_argument("GT cache r_gt disagrees with the recomputed camera-frame range");
        }
        if (rays.size() >= static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()))
        {
            throw std::invalid_argument("a frame cannot hold more than 2^32-1 rays");
        }

        Ray ray;
        ray.ray_id = (static_cast<std::uint64_t>(frame) << 32U) |
                     static_cast<std::uint64_t>(static_cast<std::uint32_t>(rays.size()));
        ray.p_L = point.p_L;
        ray.p_C = pointC.cast<float>();
        ray.dir_C = (pointC / range).cast<float>();
        ray.t_gt = static_cast<float>(range);
        ray.bucket = static_cast<std::uint8_t>(BucketOf(range));
        ray.occluded = 0U;
        ray.camera = point.camera;
        ray.semanticId = point.semanticId;
        rays.push_back(ray);
    }
    return rays;
}
} // namespace raylidar::rays
