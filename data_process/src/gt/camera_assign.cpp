#include <raylidar/gt/camera_assign.h>

#include <raylidar/overlay/project.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <string>
#include <utility>

namespace raylidar::gt
{
namespace
{
constexpr std::uint8_t kNoCamera = 255U;

Eigen::Vector3d ToCameraFrame(const Eigen::Vector3d &p_L, const Eigen::Matrix4d &T_Ck_rect_V)
{
    const Eigen::Vector4d homogeneous(p_L.x(), p_L.y(), p_L.z(), 1.0);
    return (T_Ck_rect_V * homogeneous).head<3>();
}
} // namespace

CameraEligibility EvaluateCameraEligibility(const Eigen::Vector3d &p_L,
                                            const Eigen::Matrix4d &T_Ck_rect_V,
                                            const Eigen::Matrix3d &K_rect, const int imageWidth,
                                            const int imageHeight, const double rMinM,
                                            const double rMaxM, const int marginPx)
{
    CameraEligibility eligibility;

    Eigen::MatrixXd single(1, 3);
    single << p_L.x(), p_L.y(), p_L.z();
    const overlay::ProjectionResult projected =
        overlay::ProjectPoints(single, T_Ck_rect_V, K_rect, imageWidth, imageHeight,
                               /*minRangeM=*/0.0, /*maxRangeM=*/1.0e9,
                               /*marginPx=*/0, /*clampRange=*/false);

    eligibility.z = projected.z[0];
    eligibility.rangeM = projected.rangeM[0];
    eligibility.u = projected.uv[0].x();
    eligibility.v = projected.uv[0].y();

    eligibility.zPositive =
        std::isfinite(eligibility.z) && eligibility.z > 0.0;
    const double margin = static_cast<double>(std::max(marginPx, 0));
    const double maxU = static_cast<double>(imageWidth - 1) - margin;
    const double maxV = static_cast<double>(imageHeight - 1) - margin;
    eligibility.inImageMargin =
        std::isfinite(eligibility.u) && std::isfinite(eligibility.v) &&
        eligibility.u >= margin && eligibility.u <= maxU && eligibility.v >= margin &&
        eligibility.v <= maxV;
    eligibility.rangeOk = std::isfinite(eligibility.rangeM) &&
                          eligibility.rangeM >= rMinM && eligibility.rangeM <= rMaxM;
    return eligibility;
}

CameraAssignment AssignPrimaryCamera(
    const Eigen::Vector3d &p_L, const std::map<std::string, Eigen::Matrix4d> &T_Ck_rect_V,
    const std::map<std::string, Eigen::Matrix3d> &K_rect,
    const std::map<std::string, std::pair<int, int>> &imageSizes, const double rMinM,
    const double rMaxM, const int marginPx)
{
    static const std::array<std::string, 2> kCameraOrder = {"image_00", "image_01"};

    CameraAssignment assignment;
    bool found = false;
    double bestScore = std::numeric_limits<double>::infinity();

    for (std::size_t index = 0; index < kCameraOrder.size(); ++index)
    {
        const std::string &camera = kCameraOrder[index];
        const auto transform = T_Ck_rect_V.find(camera);
        const auto intrinsics = K_rect.find(camera);
        const auto size = imageSizes.find(camera);
        if (transform == T_Ck_rect_V.end() || intrinsics == K_rect.end() ||
            size == imageSizes.end())
        {
            continue;
        }

        const CameraEligibility eligibility = EvaluateCameraEligibility(
            p_L, transform->second, intrinsics->second, size->second.first, size->second.second,
            rMinM, rMaxM, marginPx);
        if (!eligibility.valid())
        {
            continue;
        }

        const double fx = intrinsics->second(0, 0);
        const double fy = intrinsics->second(1, 1);
        const double cx = intrinsics->second(0, 2);
        const double cy = intrinsics->second(1, 2);
        const double dx = (eligibility.u - cx) / fx;
        const double dy = (eligibility.v - cy) / fy;
        const double score = dx * dx + dy * dy;
        if (!std::isfinite(score))
        {
            continue;
        }

        // Strict comparison keeps the earlier (lower-id) camera on an exact tie.
        if (!found || score < bestScore)
        {
            found = true;
            bestScore = score;
            assignment.camera = static_cast<std::uint8_t>(index);
            assignment.r_gt = eligibility.rangeM;
            assignment.u = eligibility.u;
            assignment.v = eligibility.v;
            assignment.p_C = ToCameraFrame(p_L, transform->second);
        }
    }

    if (!found)
    {
        assignment.camera = kNoCamera;
        assignment.r_gt = -1.0;
        assignment.u = 0.0;
        assignment.v = 0.0;
        assignment.p_C = Eigen::Vector3d::Zero();
    }
    return assignment;
}
} // namespace raylidar::gt
