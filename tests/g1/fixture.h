#pragma once

#include <nlohmann/json.hpp>

#include <string>
#include <string_view>

namespace raylidar::test::g1fixture
{
inline constexpr const char *kSequence = "2013_05_28_drive_0000_sync";

inline nlohmann::json GoodPreflight()
{
    nlohmann::json preflight;
    preflight["ok"] = true;
    preflight["root"] = "/data-20T/hybrid-lidar-sim/raw/kitti360";
    return preflight;
}

inline nlohmann::json GoodReadiness()
{
    nlohmann::json readiness;
    readiness["sequence"] = kSequence;
    readiness["license"] = "KITTI-360 non-commercial research use";
    readiness["image_provenance"] = "official_data_rect";
    readiness["archives"] =
        nlohmann::json::array({nlohmann::json{{"name", "calibration.zip"}, {"granularity", "all_sequences"}}});
    return readiness;
}

inline nlohmann::json GoodPrereg()
{
    nlohmann::json prereg;
    prereg["ok"] = true;
    prereg["frozen_at_utc"] = "2026-10-31T05:35:58Z";
    prereg["expected"] = "02516efe";
    prereg["actual"] = "02516efe";
    return prereg;
}

inline nlohmann::json MakeCalibReport()
{
    nlohmann::json report;
    report["schema_version"] = 1;
    report["sequence"] = kSequence;
    report["pass"] = true;
    report["max_cam0_world_delta"] = 4.7e-4;
    report["camera_height_m"] = {1.470, 1.479, 1.466};
    report["camera_height_spread_m"] = 0.013;
    report["camera_height_status"] = "ok";
    report["checks"] = {{"roundtrip", true}, {"rotation", true}, {"cam0_world", true}, {"camera_height", true}};

    nlohmann::json frames = nlohmann::json::array();
    for (const int frame : {88, 89, 90})
    {
        nlohmann::json cameras = nlohmann::json::object();
        for (const char *camera : {"image_00", "image_01"})
        {
            cameras[camera] = {{"composed", true},
                               {"roundtrip_raw", 2.4e-10},
                               {"roundtrip_rect", 1.0e-10},
                               {"rotation_orth", 9.93e-7},
                               {"rotation_det", 4.52e-7}};
        }
        frames.push_back({{"frame", frame}, {"pose_frame", frame}, {"pose_delta", 0}, {"cameras", cameras}});
    }
    report["frames"] = frames;
    return report;
}

inline nlohmann::json MakeOverlayReport()
{
    nlohmann::json report;
    report["schema_version"] = 1;
    report["sequence"] = kSequence;
    report["frames"] = {88, 89, 90};
    report["cams"] = {"image_00", "image_01"};
    report["advisory_only"] = true;
    report["rectified_data_rect_only"] = true;

    nlohmann::json results = nlohmann::json::array();
    for (const int frame : {88, 89, 90})
    {
        for (const char *camera : {"image_00", "image_01"})
        {
            const std::string cameraName(camera);
            results.push_back(
                {{"frame", frame},
                 {"camera", cameraName},
                 {"rectified", true},
                 {"image", "/data-20T/hybrid-lidar-sim/raw/kitti360/data_2d_raw/" + std::string(kSequence) + "/" +
                               cameraName + "/data_rect/0000000088.png"},
                 {"image_resolved", "/data-20T/hybrid-lidar-sim/raw/kitti360/"
                                    "data_2d_raw/" +
                                        std::string(kSequence) + "/" + cameraName + "/data_rect/0000000088.png"},
                 {"overlay_4x", "/data-20T/hybrid-lidar-sim/artifacts/01/overlays/"
                                "frame_0000000088/" +
                                    cameraName + "_overlay_4x.png"},
                 {"scale_factor", 4}});
        }
    }
    report["results"] = results;
    return report;
}

inline nlohmann::json MakeVerification()
{
    nlohmann::json verification;
    verification["schema_version"] = 1;
    verification["rectified_data_rect_only"] = true;
    verification["offset_unit"] = "original_1x_px";
    verification["reviewer"] = "test-reviewer";
    verification["measured_at_utc"] = "2026-10-31T06:00:00Z";

    nlohmann::json frames = nlohmann::json::array();
    for (const int frame : {88, 89, 90})
    {
        for (const char *camera : {"image_00", "image_01"})
        {
            frames.push_back({{"frame", frame},
                              {"camera", std::string(camera)},
                              {"features_measured", 5},
                              {"max_offset_px", 1.5},
                              {"notes", ""}});
        }
    }
    verification["frames"] = frames;
    return verification;
}

inline void SetCameraMetric(nlohmann::json &calib, const char *key, double value)
{
    for (nlohmann::json &frame : calib["frames"])
    {
        for (auto item : frame["cameras"].items())
        {
            item.value()[key] = value;
        }
    }
}

inline const nlohmann::json *FindCheck(const nlohmann::json &report, std::string_view id)
{
    if (!report.is_object() || !report.contains("checks") || !report["checks"].is_array())
    {
        return nullptr;
    }
    for (const nlohmann::json &check : report["checks"])
    {
        if (check.is_object() && check.contains("id") && check["id"] == id)
        {
            return &check;
        }
    }
    return nullptr;
}
} // namespace raylidar::test::g1fixture
