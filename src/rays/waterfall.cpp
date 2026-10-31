#include <raylidar/rays/waterfall.h>

#include <raylidar/common/io.h>
#include <raylidar/gt/camera_assign.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace raylidar::rays
{
namespace
{
constexpr int kExitGateFailed = 6;

std::int64_t ReadCount(const nlohmann::json &counts, const char *key, const std::string &context)
{
    if (!counts.is_object() || !counts.contains(key))
    {
        throw IoError(context + ": missing waterfall count '" + std::string(key) + "'",
                      kExitGateFailed);
    }
    const nlohmann::json &value = counts.at(key);
    if (value.is_number_unsigned())
    {
        const auto count = value.get<std::uint64_t>();
        if (count > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
        {
            throw IoError(context + ": waterfall count '" + std::string(key) + "' overflows",
                          kExitGateFailed);
        }
        return static_cast<std::int64_t>(count);
    }
    if (value.is_number_integer())
    {
        const auto count = value.get<std::int64_t>();
        if (count < 0)
        {
            throw IoError(context + ": waterfall count '" + std::string(key) + "' is negative",
                          kExitGateFailed);
        }
        return count;
    }
    throw IoError(context + ": waterfall count '" + std::string(key) + "' is not an integer",
                  kExitGateFailed);
}

bool IsNegative(const FrameWaterfall &waterfall)
{
    return waterfall.raw < 0 || waterfall.nonfinite < 0 || waterfall.ego_roi < 0 ||
           waterfall.dynamic_union < 0 || waterfall.no_camera_fov < 0 ||
           waterfall.range_out_of_domain < 0 || waterfall.valid < 0;
}

nlohmann::json FrameJson(const FrameWaterfall &waterfall)
{
    nlohmann::json entry = nlohmann::json::object();
    entry["frame"] = waterfall.frame;
    entry["raw"] = waterfall.raw;
    entry["valid"] = waterfall.valid;
    entry["coverage"] = waterfall.raw > 0
                            ? static_cast<double>(waterfall.valid) /
                                  static_cast<double>(waterfall.raw)
                            : 0.0;
    entry["nonfinite"] = waterfall.nonfinite;
    entry["ego_roi"] = waterfall.ego_roi;
    entry["dynamic_union"] = waterfall.dynamic_union;
    entry["no_camera_fov"] = waterfall.no_camera_fov;
    entry["range_out_of_domain"] = waterfall.range_out_of_domain;
    std::string reason;
    entry["exact"] = ValidateWaterfall(waterfall, &reason);
    if (!reason.empty())
    {
        entry["reason"] = reason;
    }
    return entry;
}
} // namespace

CameraDropClass ClassifyCamera255Point(
    const Eigen::Vector3d &p_L,
    const std::map<std::string, Eigen::Matrix4d> &T_Ck_rect_V,
    const std::map<std::string, Eigen::Matrix3d> &K_rect,
    const std::map<std::string, std::pair<int, int>> &imageSizes, const double rMinM,
    const double rMaxM, const int marginPx)
{
    static const std::array<std::string, 2> kCameraOrder = {"image_00", "image_01"};

    bool anyInFov = false;
    bool anyRangeOk = false;
    for (const std::string &camera : kCameraOrder)
    {
        const auto transform = T_Ck_rect_V.find(camera);
        const auto intrinsics = K_rect.find(camera);
        const auto size = imageSizes.find(camera);
        if (transform == T_Ck_rect_V.end() || intrinsics == K_rect.end() ||
            size == imageSizes.end())
        {
            // A camera-255 point cannot be classified without every camera.
            throw IoError("missing camera calibration or image size for " + camera,
                          kExitGateFailed);
        }

        const gt::CameraEligibility eligibility = gt::EvaluateCameraEligibility(
            p_L, transform->second, intrinsics->second, size->second.first, size->second.second,
            rMinM, rMaxM, marginPx);
        if (eligibility.valid())
        {
            throw IoError("camera-255 GT point is eligible in " + camera +
                              " (waterfall cannot double count it)",
                          kExitGateFailed);
        }
        if (eligibility.zPositive && eligibility.inImageMargin)
        {
            anyInFov = true;
            if (eligibility.rangeOk)
            {
                anyRangeOk = true;
            }
        }
    }

    if (!anyInFov)
    {
        return CameraDropClass::NoCameraFov;
    }
    // `anyRangeOk` is unreachable because an in-FOV camera with rangeOk would be
    // eligible and rejected above; keeping the branch makes the rule explicit.
    (void)anyRangeOk;
    return CameraDropClass::RangeOutOfDomain;
}

FrameWaterfall BuildFrameWaterfall(const int frame, const std::uint64_t raw,
                                   const nlohmann::json &maskCounts,
                                   const std::uint64_t keptStatic, const std::uint64_t validRays,
                                   const std::uint64_t noCameraFov,
                                   const std::uint64_t rangeOutOfDomain)
{
    const std::string context = "frame " + std::to_string(frame) + " waterfall";
    FrameWaterfall waterfall;
    waterfall.frame = frame;
    waterfall.raw = static_cast<std::int64_t>(raw);
    waterfall.nonfinite = ReadCount(maskCounts, "nonfinite", context);
    waterfall.ego_roi = ReadCount(maskCounts, "ego_roi", context);
    waterfall.dynamic_union = ReadCount(maskCounts, "dynamic_union", context);
    const std::int64_t maskKept = ReadCount(maskCounts, "kept_static", context);
    if (maskKept != static_cast<std::int64_t>(keptStatic))
    {
        throw IoError(context + ": mask kept_static disagrees with the caller", kExitGateFailed);
    }
    waterfall.valid = static_cast<std::int64_t>(validRays);
    waterfall.no_camera_fov = static_cast<std::int64_t>(noCameraFov);
    waterfall.range_out_of_domain = static_cast<std::int64_t>(rangeOutOfDomain);

    const std::uint64_t sourceSum = static_cast<std::uint64_t>(waterfall.nonfinite) +
                                    static_cast<std::uint64_t>(waterfall.ego_roi) +
                                    static_cast<std::uint64_t>(waterfall.dynamic_union) +
                                    static_cast<std::uint64_t>(maskKept);
    if (sourceSum != static_cast<std::uint64_t>(waterfall.raw))
    {
        throw IoError(context + ": raw != nonfinite + ego_roi + dynamic_union + kept_static",
                      kExitGateFailed);
    }
    const std::uint64_t keptSum = static_cast<std::uint64_t>(waterfall.valid) +
                                  static_cast<std::uint64_t>(waterfall.no_camera_fov) +
                                  static_cast<std::uint64_t>(waterfall.range_out_of_domain);
    if (keptSum != static_cast<std::uint64_t>(maskKept))
    {
        throw IoError(context + ": kept_static != valid + no_camera_fov + range_out_of_domain",
                      kExitGateFailed);
    }
    return waterfall;
}

bool ValidateWaterfall(const FrameWaterfall &waterfall, std::string *reason)
{
    if (IsNegative(waterfall))
    {
        if (reason != nullptr)
        {
            *reason = "negative waterfall count";
        }
        return false;
    }

    const std::uint64_t dropped = static_cast<std::uint64_t>(waterfall.nonfinite) +
                                  static_cast<std::uint64_t>(waterfall.ego_roi) +
                                  static_cast<std::uint64_t>(waterfall.dynamic_union) +
                                  static_cast<std::uint64_t>(waterfall.no_camera_fov) +
                                  static_cast<std::uint64_t>(waterfall.range_out_of_domain);
    const std::uint64_t expected = dropped + static_cast<std::uint64_t>(waterfall.valid);
    if (expected == static_cast<std::uint64_t>(waterfall.raw))
    {
        if (reason != nullptr)
        {
            reason->clear();
        }
        return true;
    }

    if (reason != nullptr)
    {
        const std::int64_t difference = static_cast<std::int64_t>(
            static_cast<std::uint64_t>(waterfall.raw) - expected);
        *reason = "raw=" + std::to_string(waterfall.raw) +
                  " != valid+reasons=" + std::to_string(expected) + " (difference " +
                  std::to_string(difference) + ")";
    }
    return false;
}

nlohmann::json AggregateWaterfalls(const std::vector<FrameWaterfall> &frames)
{
    FrameWaterfall totals;
    totals.frame = -1;
    bool allExact = true;
    nlohmann::json frameArray = nlohmann::json::array();
    for (const FrameWaterfall &frame : frames)
    {
        frameArray.push_back(FrameJson(frame));
        totals.raw += frame.raw;
        totals.nonfinite += frame.nonfinite;
        totals.ego_roi += frame.ego_roi;
        totals.dynamic_union += frame.dynamic_union;
        totals.no_camera_fov += frame.no_camera_fov;
        totals.range_out_of_domain += frame.range_out_of_domain;
        totals.valid += frame.valid;
        std::string reason;
        if (!ValidateWaterfall(frame, &reason))
        {
            allExact = false;
        }
    }

    nlohmann::json document = nlohmann::json::object();
    document["schema_version"] = 1;
    document["frames"] = std::move(frameArray);
    document["totals"] = FrameJson(totals);
    document["totals"]["exact"] = allExact && ValidateWaterfall(totals, nullptr);
    document["exact"] = allExact && ValidateWaterfall(totals, nullptr);
    return document;
}
} // namespace raylidar::rays
