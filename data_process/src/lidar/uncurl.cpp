#include <raylidar/common/io.h>
#include <raylidar/lidar/uncurl.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace raylidar::lidar
{
namespace
{
constexpr int kExitInvalidInput = 4;
constexpr double kTwoPi = 6.283185307179586476925286766559;
constexpr double kAzimuthClamp = 1.0 - 1e-12;

void RequireFinite(const Eigen::Matrix4d &transform, const char *what)
{
    if (!transform.allFinite())
    {
        throw std::invalid_argument(std::string(what) + " is not finite");
    }
}
} // namespace

double AzimuthFraction(const Eigen::Vector3d &p)
{
    if (!p.allFinite())
    {
        throw std::invalid_argument("AzimuthFraction: point is not finite");
    }
    double u = 0.5 + std::atan2(p.y(), p.x()) / kTwoPi;
    if (u < 0.0)
    {
        u = 0.0;
    }
    if (u >= 1.0)
    {
        u = kAzimuthClamp;
    }
    return u;
}

Eigen::Matrix4d InterpolateRigid(const Eigen::Matrix4d &T_a, const Eigen::Matrix4d &T_b, const double u)
{
    RequireFinite(T_a, "T_a");
    RequireFinite(T_b, "T_b");
    if (!std::isfinite(u))
    {
        throw std::invalid_argument("InterpolateRigid: u is not finite");
    }

    const Eigen::Matrix4d relative = T_a.inverse() * T_b;
    const Eigen::Matrix3d rotation = relative.block<3, 3>(0, 0);
    const Eigen::Vector3d translation = relative.block<3, 1>(0, 3);

    const Eigen::AngleAxisd angleAxis(rotation);
    const double angle = angleAxis.angle();
    const Eigen::Vector3d axis = angleAxis.axis();

    Eigen::Matrix3d scaledRotation = Eigen::Matrix3d::Identity();
    if (std::abs(angle) > 1e-12)
    {
        scaledRotation = Eigen::AngleAxisd(angle * u, axis).toRotationMatrix();
    }

    Eigen::Matrix4d delta = Eigen::Matrix4d::Identity();
    delta.block<3, 3>(0, 0) = scaledRotation;
    delta.block<3, 1>(0, 3) = u * translation;
    return T_a * delta;
}

Eigen::Vector3d UncurlPoint(const Eigen::Vector3d &p_curr, const UncurlParams &params)
{
    if (!p_curr.allFinite())
    {
        throw std::invalid_argument("UncurlPoint: point is not finite");
    }
    RequireFinite(params.T_W_V_curr, "T_W_V_curr");
    RequireFinite(params.T_W_V_next, "T_W_V_next");

    const double u = AzimuthFraction(p_curr);
    const Eigen::Matrix4d T_u = InterpolateRigid(params.T_W_V_curr, params.T_W_V_next, u);
    const Eigen::Vector4d homogeneous(p_curr.x(), p_curr.y(), p_curr.z(), 1.0);
    const Eigen::Vector4d reference = params.T_W_V_curr.inverse() * T_u * homogeneous;
    return reference.head<3>();
}

std::vector<Eigen::Vector3d> UncurlScan(const std::vector<PointXYZI> &points, const UncurlParams &params,
                                        std::size_t *rejectedNonFinite)
{
    if (params.t_next_ns <= params.t_curr_ns)
    {
        throw IoError("uncurl sweep interval is not strictly positive: [" + std::to_string(params.t_curr_ns) + ", " +
                          std::to_string(params.t_next_ns) + ")",
                      kExitInvalidInput);
    }

    std::vector<Eigen::Vector3d> uncurled;
    uncurled.reserve(points.size());
    std::size_t rejected = 0;
    for (const PointXYZI &point : points)
    {
        const Eigen::Vector3d p(point.x, point.y, point.z);
        if (!p.allFinite())
        {
            ++rejected;
            continue;
        }
        uncurled.push_back(UncurlPoint(p, params));
    }

    if (rejectedNonFinite != nullptr)
    {
        *rejectedNonFinite = rejected;
    }
    return uncurled;
}
} // namespace raylidar::lidar
