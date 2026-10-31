#include "fixture.h"

#include <raylidar/g1/g1.h>

#include <catch2/catch_test_macros.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <string>
#include <string_view>
#include <vector>

#ifndef RAYLIDAR_SOURCE_DIR
#define RAYLIDAR_SOURCE_DIR "."
#endif

namespace
{
using raylidar::g1::EvaluateG1;
using raylidar::g1::ExpectedPairsFromOverlay;
using raylidar::g1::RenderMarkdown;
using raylidar::g1::Thresholds;
using raylidar::g1::ValidateOverlayVerification;
using raylidar::test::g1fixture::FindCheck;
using raylidar::test::g1fixture::GoodPrereg;
using raylidar::test::g1fixture::GoodPreflight;
using raylidar::test::g1fixture::GoodReadiness;
using raylidar::test::g1fixture::MakeCalibReport;
using raylidar::test::g1fixture::MakeOverlayReport;
using raylidar::test::g1fixture::MakeVerification;
using raylidar::test::g1fixture::SetCameraMetric;

nlohmann::json Evaluate(const nlohmann::json &calib, const nlohmann::json &overlay,
                        const nlohmann::json *verification)
{
    return EvaluateG1(GoodPreflight(), GoodReadiness(), calib, overlay, GoodPrereg(),
                      verification);
}

std::string CheckStatus(const nlohmann::json &report, std::string_view id)
{
    const nlohmann::json *check = FindCheck(report, id);
    REQUIRE(check != nullptr);
    return check->at("status").get<std::string>();
}

int CountContaining(const nlohmann::json &values, std::string_view needle)
{
    int count = 0;
    if (!values.is_array())
    {
        return count;
    }
    for (const nlohmann::json &value : values)
    {
        if (!value.is_string())
        {
            continue;
        }
        if (value.get<std::string>().find(needle) != std::string::npos)
        {
            ++count;
        }
    }
    return count;
}
} // namespace

TEST_CASE("matching synthetic reports pass G1", "[g1]")
{
    const nlohmann::json verification = MakeVerification();
    const nlohmann::json report =
        Evaluate(MakeCalibReport(), MakeOverlayReport(), &verification);

    CHECK(report["schema_version"] == 1);
    CHECK(report["pass"] == true);
    CHECK(report["checks"].size() == 10U);
    CHECK(report["blocking_failures"].empty());

    for (const char *id : {"preflight_ready", "readiness_complete", "decisions_verified",
                           "transform_roundtrip", "rotation_orthonormal", "cam0_world",
                           "camera_height", "overlay_frames",
                           "rectified_data_rect_only", "overlay_offsets"})
    {
        CHECK(CheckStatus(report, id) == "pass");
    }

    const std::string markdown = RenderMarkdown(report);
    CHECK(markdown.find("G1 PASSED") != std::string::npos);
    CHECK(markdown.find("G1 NOT PASSED") == std::string::npos);
}

TEST_CASE("cam0_to_world deltas above 1e-3 fail", "[g1]")
{
    nlohmann::json calib = MakeCalibReport();
    calib["max_cam0_world_delta"] = 2e-3;
    const nlohmann::json verification = MakeVerification();
    const nlohmann::json report = Evaluate(calib, MakeOverlayReport(), &verification);

    CHECK(CheckStatus(report, "cam0_world") == "fail");
    CHECK(report["pass"] == false);
}

TEST_CASE("rotation tolerances use official-file precision", "[g1]")
{
    SECTION("9.93e-7 orthonormality deviation passes")
    {
        nlohmann::json calib = MakeCalibReport();
        SetCameraMetric(calib, "rotation_orth", 9.93e-7);
        SetCameraMetric(calib, "rotation_det", 4.52e-7);
        const nlohmann::json verification = MakeVerification();
        const nlohmann::json report = Evaluate(calib, MakeOverlayReport(), &verification);
        CHECK(CheckStatus(report, "rotation_orthonormal") == "pass");
    }

    SECTION("1e-3 deviation fails")
    {
        nlohmann::json calib = MakeCalibReport();
        SetCameraMetric(calib, "rotation_orth", 1e-3);
        SetCameraMetric(calib, "rotation_det", 1e-3);
        const nlohmann::json verification = MakeVerification();
        const nlohmann::json report = Evaluate(calib, MakeOverlayReport(), &verification);
        CHECK(CheckStatus(report, "rotation_orthonormal") == "fail");
        CHECK(report["pass"] == false);
    }
}

TEST_CASE("world-only round-trip evidence never satisfies transform_roundtrip", "[g1]")
{
    nlohmann::json calib = MakeCalibReport();
    for (nlohmann::json &frame : calib["frames"])
    {
        for (auto item : frame["cameras"].items())
        {
            item.value().erase("roundtrip_raw");
            item.value().erase("roundtrip_rect");
        }
    }
    calib["roundtrip_world"] = 1e-12;

    const nlohmann::json verification = MakeVerification();
    const nlohmann::json report = Evaluate(calib, MakeOverlayReport(), &verification);
    const std::string status = CheckStatus(report, "transform_roundtrip");
    CHECK(status != "pass");
    CHECK((status == "fail" || status == "missing"));
    CHECK(report["pass"] == false);
}

TEST_CASE("camera height band and status are enforced", "[g1]")
{
    SECTION("height outside [1.4, 1.9] fails")
    {
        nlohmann::json calib = MakeCalibReport();
        calib["camera_height_m"] = {1.95, 1.96, 1.97};
        const nlohmann::json verification = MakeVerification();
        const nlohmann::json report = Evaluate(calib, MakeOverlayReport(), &verification);
        CHECK(CheckStatus(report, "camera_height") == "fail");
    }

    SECTION("skipped_no_lidar fails")
    {
        nlohmann::json calib = MakeCalibReport();
        calib["camera_height_status"] = "skipped_no_lidar";
        const nlohmann::json verification = MakeVerification();
        const nlohmann::json report = Evaluate(calib, MakeOverlayReport(), &verification);
        CHECK(CheckStatus(report, "camera_height") == "fail");
    }
}

TEST_CASE("overlay needs at least three unique frames", "[g1]")
{
    nlohmann::json overlay = MakeOverlayReport();
    overlay["frames"] = {88, 89};
    nlohmann::json results = nlohmann::json::array();
    for (const nlohmann::json &result : overlay["results"])
    {
        if (result["frame"] == 88 || result["frame"] == 89)
        {
            results.push_back(result);
        }
    }
    overlay["results"] = results;

    nlohmann::json verification = MakeVerification();
    nlohmann::json entries = nlohmann::json::array();
    for (const nlohmann::json &entry : verification["frames"])
    {
        if (entry["frame"] == 88 || entry["frame"] == 89)
        {
            entries.push_back(entry);
        }
    }
    verification["frames"] = entries;

    const nlohmann::json report = Evaluate(MakeCalibReport(), overlay, &verification);
    CHECK(CheckStatus(report, "overlay_frames") == "fail");
    CHECK(report["pass"] == false);
}

TEST_CASE("offsets of exactly 2.0 px fail the strict gate", "[g1]")
{
    nlohmann::json verification = MakeVerification();
    for (nlohmann::json &entry : verification["frames"])
    {
        entry["max_offset_px"] = 2.0;
    }
    const nlohmann::json report =
        Evaluate(MakeCalibReport(), MakeOverlayReport(), &verification);
    CHECK(CheckStatus(report, "overlay_offsets") == "fail");
    CHECK(report["pass"] == false);
}

TEST_CASE("fewer than five measured features is a schema error", "[g1]")
{
    nlohmann::json verification = MakeVerification();
    verification["frames"][0]["features_measured"] = 4;

    const auto expected = ExpectedPairsFromOverlay(MakeOverlayReport());
    const std::vector<std::string> errors =
        ValidateOverlayVerification(verification, &expected);
    CHECK(!errors.empty());

    const nlohmann::json report =
        Evaluate(MakeCalibReport(), MakeOverlayReport(), &verification);
    CHECK(CheckStatus(report, "overlay_offsets") == "fail");
}

TEST_CASE("verification must cover every expected pair", "[g1]")
{
    SECTION("one covered pair leaves five missing pairs")
    {
        nlohmann::json verification = MakeVerification();
        verification["frames"] = nlohmann::json::array({verification["frames"][0]});

        const nlohmann::json report =
            Evaluate(MakeCalibReport(), MakeOverlayReport(), &verification);
        CHECK(CheckStatus(report, "overlay_offsets") == "fail");
        CHECK(report["pass"] == false);
        CHECK(CountContaining(report["blocking_failures"], "missing pair") == 5);
    }

    SECTION("duplicate pair fails")
    {
        nlohmann::json verification = MakeVerification();
        verification["frames"].push_back(verification["frames"][0]);

        const nlohmann::json report =
            Evaluate(MakeCalibReport(), MakeOverlayReport(), &verification);
        CHECK(CheckStatus(report, "overlay_offsets") == "fail");
        CHECK(CountContaining(report["blocking_failures"], "duplicate pair") == 1);
    }
}

TEST_CASE("ExpectedPairsFromOverlay returns the full cross product", "[g1]")
{
    const auto pairs = ExpectedPairsFromOverlay(MakeOverlayReport());
    REQUIRE(pairs.size() == 6U);
    CHECK(pairs[0] == std::make_pair(88, std::string("image_00")));
    CHECK(pairs[1] == std::make_pair(88, std::string("image_01")));
    CHECK(pairs[2] == std::make_pair(89, std::string("image_00")));
    CHECK(pairs[3] == std::make_pair(89, std::string("image_01")));
    CHECK(pairs[4] == std::make_pair(90, std::string("image_00")));
    CHECK(pairs[5] == std::make_pair(90, std::string("image_01")));
}

TEST_CASE("only rectified data_rect results are accepted", "[g1]")
{
    SECTION("rectified false fails")
    {
        nlohmann::json overlay = MakeOverlayReport();
        overlay["results"][0]["rectified"] = false;
        const nlohmann::json verification = MakeVerification();
        const nlohmann::json report =
            Evaluate(MakeCalibReport(), overlay, &verification);
        CHECK(CheckStatus(report, "rectified_data_rect_only") == "fail");
    }

    SECTION("image outside data_rect fails")
    {
        nlohmann::json overlay = MakeOverlayReport();
        overlay["results"][0]["image"] =
            "/data/data_2d_raw/seq/image_00/data_rgb/0000000088.png";
        const nlohmann::json verification = MakeVerification();
        const nlohmann::json report =
            Evaluate(MakeCalibReport(), overlay, &verification);
        CHECK(CheckStatus(report, "rectified_data_rect_only") == "fail");
    }
}

TEST_CASE("verification must declare original 1x pixel units", "[g1]")
{
    nlohmann::json verification = MakeVerification();
    verification.erase("offset_unit");

    const auto expected = ExpectedPairsFromOverlay(MakeOverlayReport());
    const std::vector<std::string> errors =
        ValidateOverlayVerification(verification, &expected);
    CHECK(!errors.empty());

    const nlohmann::json report =
        Evaluate(MakeCalibReport(), MakeOverlayReport(), &verification);
    CHECK(CheckStatus(report, "overlay_offsets") == "fail");
}

TEST_CASE("missing verification reports pending_human and never passes", "[g1]")
{
    const nlohmann::json report =
        Evaluate(MakeCalibReport(), MakeOverlayReport(), nullptr);
    CHECK(CheckStatus(report, "overlay_offsets") == "pending_human");
    CHECK(report["pass"] == false);

    const std::string markdown = RenderMarkdown(report);
    CHECK(markdown.find("G1 NOT PASSED") != std::string::npos);
    CHECK(markdown.find("G1 PASSED") == std::string::npos);
}

TEST_CASE("tampered decisions fail decisions_verified", "[g1]")
{
    nlohmann::json prereg = GoodPrereg();
    prereg["ok"] = false;
    const nlohmann::json verification = MakeVerification();
    const nlohmann::json report = EvaluateG1(GoodPreflight(), GoodReadiness(),
                                             MakeCalibReport(), MakeOverlayReport(),
                                             prereg, &verification);
    CHECK(CheckStatus(report, "decisions_verified") == "fail");
    CHECK(report["pass"] == false);
}

TEST_CASE("G1Thresholds match configs/frames.yaml", "[g1]")
{
    const YAML::Node config =
        YAML::LoadFile(std::string(RAYLIDAR_SOURCE_DIR) + "/configs/frames.yaml");
    const YAML::Node assertions = config["assertions"];
    const YAML::Node overlay = config["overlay"];
    const auto &thresholds = Thresholds();

    CHECK(thresholds.roundtripAtol == assertions["roundtrip_atol"].as<double>());
    CHECK(thresholds.rotationAtol == assertions["rotation_atol"].as<double>());
    CHECK(thresholds.rotationDetAtol == assertions["rotation_det_atol"].as<double>());
    CHECK(thresholds.cam0WorldAtol == assertions["cam0_world_atol"].as<double>());
    CHECK(thresholds.cameraHeightMinM == assertions["camera_height_m"][0].as<double>());
    CHECK(thresholds.cameraHeightMaxM == assertions["camera_height_m"][1].as<double>());
    CHECK(thresholds.cameraHeightStabilityM ==
          assertions["camera_height_stability_m"].as<double>());
    CHECK(thresholds.overlayMinFrames == overlay["min_frames"].as<int>());
    CHECK(thresholds.overlayMinFeatures == overlay["min_features"].as<int>());
    CHECK(thresholds.overlayMaxOffsetPx == overlay["max_offset_px"].as<double>());
}
