#include <raylidar/g1/g1.h>

#include <raylidar/common/io.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace raylidar::g1
{
namespace
{
namespace fs = std::filesystem;

constexpr const char *kPass = "pass";
constexpr const char *kFail = "fail";
constexpr const char *kMissing = "missing";
constexpr const char *kPendingHuman = "pending_human";

constexpr const char *kRectifiedFlag = "rectified_data_rect_only";
constexpr const char *kOffsetUnit = "original_1x_px";
constexpr const char *kAttestationKey = "attestation";
constexpr const char *kAttestationAllBelow = "all_offsets_below_threshold";
constexpr const char *kThresholdPxKey = "threshold_px";

// Hard thresholds live here as code constants. `Thresholds()` returns this
// object and nothing can replace it.
constexpr G1Thresholds kThresholds{};

bool GetBool(const nlohmann::json &object, const char *key, bool fallback)
{
    if (!object.is_object())
    {
        return fallback;
    }
    const auto found = object.find(key);
    if (found == object.end() || !found->is_boolean())
    {
        return fallback;
    }
    return found->get<bool>();
}

std::string GetString(const nlohmann::json &object, const char *key,
                      const std::string &fallback = {})
{
    if (!object.is_object())
    {
        return fallback;
    }
    const auto found = object.find(key);
    if (found == object.end() || !found->is_string())
    {
        return fallback;
    }
    return found->get<std::string>();
}

int GetInt(const nlohmann::json &object, const char *key, int fallback)
{
    if (!object.is_object())
    {
        return fallback;
    }
    const auto found = object.find(key);
    if (found == object.end() || !found->is_number_integer())
    {
        return fallback;
    }
    return found->get<int>();
}

bool TryGetNumber(const nlohmann::json &value, double &out)
{
    if (!value.is_number())
    {
        return false;
    }
    const double number = value.get<double>();
    if (!std::isfinite(number))
    {
        return false;
    }
    out = number;
    return true;
}

bool HasAllBelowAttestation(const nlohmann::json &object)
{
    if (!object.is_object())
    {
        return false;
    }
    const auto found = object.find(kAttestationKey);
    return found != object.end() && found->is_string() &&
           found->get<std::string>() == kAttestationAllBelow;
}

std::string PairLabel(int frame, const std::string &camera)
{
    return "(" + std::to_string(frame) + ", " + camera + ")";
}

bool IsUtcTimestamp(const std::string &value)
{
    static const std::regex pattern(R"(^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$)");
    return std::regex_match(value, pattern);
}

bool HasPathComponent(const std::string &pathText, const std::string &component)
{
    if (pathText.empty())
    {
        return false;
    }
    for (const fs::path &part : fs::path(pathText))
    {
        if (part.string() == component)
        {
            return true;
        }
    }
    return false;
}

nlohmann::json MakeCheck(const std::string &id, const std::string &status,
                         nlohmann::json measured, nlohmann::json threshold,
                         nlohmann::json evidence)
{
    nlohmann::json check;
    check["id"] = id;
    check["status"] = status;
    check["measured"] = std::move(measured);
    check["threshold"] = std::move(threshold);
    check["evidence"] = std::move(evidence);
    return check;
}

/// One metric gathered from every `frames[].cameras[cam][key]` entry.
struct MetricScan
{
    bool complete = true;
    bool present = false;
    double maximum = 0.0;
    int count = 0;
};

MetricScan ScanCameraKeys(const nlohmann::json &calibReport, const char *keyA,
                          const char *keyB)
{
    MetricScan scan;
    if (!calibReport.is_object())
    {
        scan.complete = false;
        return scan;
    }
    const auto frames = calibReport.find("frames");
    if (frames == calibReport.end() || !frames->is_array())
    {
        scan.complete = false;
        return scan;
    }
    for (const nlohmann::json &frame : *frames)
    {
        if (!frame.is_object())
        {
            scan.complete = false;
            continue;
        }
        const auto cameras = frame.find("cameras");
        if (cameras == frame.end() || !cameras->is_object() || cameras->empty())
        {
            scan.complete = false;
            continue;
        }
        for (auto camera = cameras->begin(); camera != cameras->end(); ++camera)
        {
            const nlohmann::json &values = camera.value();
            if (!values.is_object())
            {
                scan.complete = false;
                continue;
            }
            double valueA = 0.0;
            double valueB = 0.0;
            const auto foundA = values.find(keyA);
            const auto foundB = values.find(keyB);
            if (foundA == values.end() || foundB == values.end() ||
                !TryGetNumber(foundA.value(), valueA) ||
                !TryGetNumber(foundB.value(), valueB))
            {
                scan.complete = false;
                continue;
            }
            scan.present = true;
            scan.maximum = std::max(scan.maximum, std::max(valueA, valueB));
            ++scan.count;
        }
    }
    if (scan.count == 0)
    {
        scan.complete = false;
    }
    return scan;
}

std::string StatusFor(bool pass, bool present)
{
    if (!present)
    {
        return kMissing;
    }
    return pass ? kPass : kFail;
}

std::string EscapeCell(std::string text)
{
    for (char &character : text)
    {
        if (character == '\n' || character == '\r')
        {
            character = ' ';
        }
    }
    std::string escaped;
    escaped.reserve(text.size());
    for (const char character : text)
    {
        if (character == '|')
        {
            escaped += "\\|";
        }
        else
        {
            escaped += character;
        }
    }
    return escaped;
}

std::string FormatValue(const nlohmann::json &value)
{
    if (value.is_null())
    {
        return "-";
    }
    if (value.is_string())
    {
        return EscapeCell(value.get<std::string>());
    }
    if (value.is_boolean())
    {
        return value.get<bool>() ? "true" : "false";
    }
    if (value.is_number())
    {
        std::ostringstream out;
        out << std::setprecision(12) << value.get<double>();
        return out.str();
    }
    return EscapeCell(value.dump());
}

std::string FormatEvidence(const nlohmann::json &value)
{
    if (!value.is_array())
    {
        return FormatValue(value);
    }
    std::string joined;
    for (const nlohmann::json &item : value)
    {
        if (!joined.empty())
        {
            joined += "; ";
        }
        joined += FormatValue(item);
    }
    return joined.empty() ? "-" : joined;
}

const nlohmann::json *FindCheck(const nlohmann::json &report, const char *id)
{
    if (!report.is_object())
    {
        return nullptr;
    }
    const auto checks = report.find("checks");
    if (checks == report.end() || !checks->is_array())
    {
        return nullptr;
    }
    for (const nlohmann::json &check : *checks)
    {
        if (GetString(check, "id") == id)
        {
            return &check;
        }
    }
    return nullptr;
}
} // namespace

const G1Thresholds &Thresholds()
{
    return kThresholds;
}

std::vector<std::string> ValidateOverlayVerification(
    const nlohmann::json &obj, const std::vector<std::pair<int, std::string>> *expected)
{
    std::vector<std::string> errors;
    if (!obj.is_object())
    {
        errors.emplace_back("verification is not a JSON object");
        return errors;
    }

    const auto schema = obj.find("schema_version");
    if (schema == obj.end() || !schema->is_number_integer() || schema->get<int>() != 1)
    {
        errors.emplace_back("schema_version must be 1");
    }

    const auto rectified = obj.find(kRectifiedFlag);
    if (rectified == obj.end() || !rectified->is_boolean() || !rectified->get<bool>())
    {
        errors.emplace_back("rectified_data_rect_only must be true");
    }

    const auto unit = obj.find("offset_unit");
    if (unit == obj.end() || !unit->is_string() ||
        unit->get<std::string>() != kOffsetUnit)
    {
        errors.emplace_back("offset_unit must be \"original_1x_px\"");
    }

    const auto reviewer = obj.find("reviewer");
    if (reviewer == obj.end() || !reviewer->is_string() ||
        reviewer->get<std::string>().empty())
    {
        errors.emplace_back("reviewer must be a non-empty string");
    }

    const auto measuredAt = obj.find("measured_at_utc");
    if (measuredAt == obj.end() || !measuredAt->is_string() ||
        !IsUtcTimestamp(measuredAt->get<std::string>()))
    {
        errors.emplace_back("measured_at_utc must match YYYY-MM-DDTHH:MM:SSZ");
    }

    const bool attestationMode = HasAllBelowAttestation(obj);
    const auto attestationIt = obj.find(kAttestationKey);
    if (attestationIt != obj.end() &&
        (!attestationIt->is_string() ||
         attestationIt->get<std::string>() != kAttestationAllBelow))
    {
        errors.emplace_back("attestation must be \"all_offsets_below_threshold\"");
    }
    if (attestationMode)
    {
        const auto thresholdIt = obj.find(kThresholdPxKey);
        double threshold = 0.0;
        if (thresholdIt == obj.end() || !TryGetNumber(*thresholdIt, threshold) ||
            std::abs(threshold - kThresholds.overlayMaxOffsetPx) > 1e-9)
        {
            errors.emplace_back("threshold_px must equal " +
                                FormatValue(kThresholds.overlayMaxOffsetPx) +
                                " when attestation is used");
        }
    }

    const auto framesIt = obj.find("frames");
    if (framesIt == obj.end() || !framesIt->is_array())
    {
        errors.emplace_back("frames must be an array");
        return errors;
    }

    std::map<std::pair<int, std::string>, int> counts;
    const nlohmann::json &frames = *framesIt;
    for (std::size_t index = 0; index < frames.size(); ++index)
    {
        const nlohmann::json &entry = frames[index];
        const std::string prefix = "frames[" + std::to_string(index) + "]: ";
        if (!entry.is_object())
        {
            errors.push_back(prefix + "entry must be an object");
            continue;
        }

        bool frameOk = false;
        int frame = 0;
        const auto frameIt = entry.find("frame");
        if (frameIt != entry.end() && frameIt->is_number_integer())
        {
            frame = frameIt->get<int>();
            frameOk = true;
        }
        else
        {
            errors.push_back(prefix + "frame must be an integer");
        }

        bool cameraOk = false;
        std::string camera;
        const auto cameraIt = entry.find("camera");
        if (cameraIt != entry.end() && cameraIt->is_string() &&
            !cameraIt->get<std::string>().empty())
        {
            camera = cameraIt->get<std::string>();
            cameraOk = true;
        }
        else
        {
            errors.push_back(prefix + "camera must be a non-empty string");
        }

        const auto featuresIt = entry.find("features_measured");
        if (featuresIt == entry.end() || !featuresIt->is_number_integer() ||
            featuresIt->get<int>() < kThresholds.overlayMinFeatures)
        {
            errors.push_back(prefix + "features_measured must be an integer >= " +
                             std::to_string(kThresholds.overlayMinFeatures));
        }

        const auto offsetIt = entry.find("max_offset_px");
        const bool attestedPair =
            attestationMode && (offsetIt == entry.end() || offsetIt->is_null());
        if (!attestedPair)
        {
            double offset = 0.0;
            if (offsetIt == entry.end() || !TryGetNumber(*offsetIt, offset) ||
                offset < 0.0)
            {
                errors.push_back(prefix + "max_offset_px must be a finite number >= 0");
            }
        }

        const auto notesIt = entry.find("notes");
        if (notesIt == entry.end() || !notesIt->is_string())
        {
            errors.push_back(prefix + "notes must be a string");
        }

        if (frameOk && cameraOk)
        {
            ++counts[{frame, camera}];
        }
    }

    if (expected != nullptr)
    {
        for (const auto &pair : *expected)
        {
            const auto found = counts.find(pair);
            const int count = found == counts.end() ? 0 : found->second;
            if (count == 0)
            {
                errors.push_back("missing pair " + PairLabel(pair.first, pair.second));
            }
            else if (count > 1)
            {
                errors.push_back("duplicate pair " + PairLabel(pair.first, pair.second));
            }
        }
        for (const auto &[pair, count] : counts)
        {
            (void)count;
            if (std::find(expected->begin(), expected->end(), pair) == expected->end())
            {
                errors.push_back("unexpected pair " + PairLabel(pair.first, pair.second));
            }
        }
    }

    return errors;
}

std::vector<std::pair<int, std::string>> ExpectedPairsFromOverlay(
    const nlohmann::json &overlayReport)
{
    if (!overlayReport.is_object())
    {
        throw std::invalid_argument("overlay report is not a JSON object");
    }

    const auto framesIt = overlayReport.find("frames");
    if (framesIt == overlayReport.end() || !framesIt->is_array())
    {
        throw std::invalid_argument("overlay report frames must be an array");
    }
    const auto camsIt = overlayReport.find("cams");
    if (camsIt == overlayReport.end() || !camsIt->is_array())
    {
        throw std::invalid_argument("overlay report cams must be an array");
    }

    std::vector<int> frames;
    for (const nlohmann::json &value : *framesIt)
    {
        if (!value.is_number_integer())
        {
            throw std::invalid_argument("overlay report frames must contain integers");
        }
        const int frame = value.get<int>();
        if (std::find(frames.begin(), frames.end(), frame) == frames.end())
        {
            frames.push_back(frame);
        }
    }
    std::vector<std::string> cams;
    for (const nlohmann::json &value : *camsIt)
    {
        if (!value.is_string())
        {
            throw std::invalid_argument("overlay report cams must contain strings");
        }
        const std::string camera = value.get<std::string>();
        if (std::find(cams.begin(), cams.end(), camera) == cams.end())
        {
            cams.push_back(camera);
        }
    }
    if (frames.empty() || cams.empty())
    {
        throw std::invalid_argument("overlay report has an empty frames or cams set");
    }

    std::vector<std::pair<int, std::string>> expected;
    expected.reserve(frames.size() * cams.size());
    for (const int frame : frames)
    {
        for (const std::string &camera : cams)
        {
            expected.emplace_back(frame, camera);
        }
    }

    const auto resultsIt = overlayReport.find("results");
    if (resultsIt == overlayReport.end() || !resultsIt->is_array())
    {
        throw std::invalid_argument("overlay report results must be an array");
    }

    std::set<std::pair<int, std::string>> seen;
    for (const nlohmann::json &result : *resultsIt)
    {
        if (!result.is_object())
        {
            throw std::invalid_argument("overlay report result entries must be objects");
        }
        const auto frameIt = result.find("frame");
        const auto cameraIt = result.find("camera");
        if (frameIt == result.end() || !frameIt->is_number_integer() ||
            cameraIt == result.end() || !cameraIt->is_string())
        {
            throw std::invalid_argument("overlay result is missing a frame or camera");
        }
        const int frame = frameIt->get<int>();
        const std::string camera = cameraIt->get<std::string>();
        if (std::find(frames.begin(), frames.end(), frame) == frames.end() ||
            std::find(cams.begin(), cams.end(), camera) == cams.end())
        {
            throw std::invalid_argument("overlay result references pair " +
                                        PairLabel(frame, camera) +
                                        " absent from frames/cams");
        }
        seen.emplace(frame, camera);
    }

    for (const auto &pair : expected)
    {
        if (seen.find(pair) == seen.end())
        {
            throw std::invalid_argument("overlay report has no result for pair " +
                                        PairLabel(pair.first, pair.second));
        }
    }

    return expected;
}

nlohmann::json EvaluateG1(const nlohmann::json &preflight, const nlohmann::json &readiness,
                          const nlohmann::json &calibReport,
                          const nlohmann::json &overlayReport, const nlohmann::json &prereg,
                          const nlohmann::json *verification)
{
    nlohmann::json checks = nlohmann::json::array();

    // preflight_ready
    {
        const bool ok = GetBool(preflight, "ok", false);
        nlohmann::json evidence;
        evidence["root"] = GetString(preflight, "root");
        checks.push_back(MakeCheck("preflight_ready", ok ? kPass : kFail, ok, true,
                                   std::move(evidence)));
    }

    // readiness_complete
    {
        const std::string sequence = GetString(readiness, "sequence");
        const std::string calibSequence = GetString(calibReport, "sequence");
        const std::string overlaySequence = GetString(overlayReport, "sequence");
        const std::string license = GetString(readiness, "license");
        const std::string provenance = GetString(readiness, "image_provenance");
        const int archiveCount =
            readiness.is_object() && readiness.contains("archives") &&
                    readiness["archives"].is_array()
                ? static_cast<int>(readiness["archives"].size())
                : 0;

        std::vector<std::string> reasons;
        if (sequence.empty())
        {
            reasons.emplace_back("readiness.sequence is missing");
        }
        if (!calibSequence.empty() && calibSequence != sequence)
        {
            reasons.emplace_back("calib sequence '" + calibSequence +
                                 "' does not match readiness sequence '" + sequence + "'");
        }
        if (!overlaySequence.empty() && overlaySequence != sequence)
        {
            reasons.emplace_back("overlay sequence '" + overlaySequence +
                                 "' does not match readiness sequence '" + sequence + "'");
        }
        if (license.empty())
        {
            reasons.emplace_back("readiness.license is missing");
        }
        if (provenance.empty())
        {
            reasons.emplace_back("readiness.image_provenance is missing");
        }
        if (archiveCount == 0)
        {
            reasons.emplace_back("readiness.archives is empty");
        }

        nlohmann::json measured;
        measured["sequence"] = sequence;
        measured["calib_sequence"] = calibSequence;
        measured["overlay_sequence"] = overlaySequence;
        measured["license"] = license;
        measured["image_provenance"] = provenance;
        measured["archives"] = archiveCount;

        nlohmann::json threshold;
        threshold["sequence_matches"] = true;
        threshold["license_present"] = true;
        threshold["provenance_present"] = true;
        threshold["archives_nonempty"] = true;

        checks.push_back(MakeCheck("readiness_complete",
                                   reasons.empty() ? kPass : kFail, std::move(measured),
                                   std::move(threshold), reasons));
    }

    // decisions_verified
    {
        const bool ok = GetBool(prereg, "ok", false);
        std::string frozenAt = GetString(prereg, "frozen_at_utc");
        if (frozenAt.empty())
        {
            frozenAt = GetString(prereg, "timestamp");
        }
        const bool stamped = !frozenAt.empty() && frozenAt != "PENDING";
        const bool passed = ok && stamped;

        nlohmann::json measured;
        measured["ok"] = ok;
        measured["frozen_at_utc"] = frozenAt;
        measured["expected"] = GetString(prereg, "expected");
        measured["actual"] = GetString(prereg, "actual");

        nlohmann::json threshold;
        threshold["ok"] = true;
        threshold["frozen_at_utc"] = "not PENDING";

        checks.push_back(MakeCheck("decisions_verified", passed ? kPass : kFail,
                                   std::move(measured), std::move(threshold),
                                   "decisions hash verified at G1 time"));
    }

    // transform_roundtrip
    {
        const bool reportPass = GetBool(calibReport, "pass", false);
        const MetricScan scan =
            ScanCameraKeys(calibReport, "roundtrip_raw", "roundtrip_rect");
        const bool within =
            scan.present && scan.maximum <= kThresholds.roundtripAtol;
        std::string status = kMissing;
        if (!scan.present || !scan.complete)
        {
            status = kMissing;
        }
        else if (!reportPass || !within)
        {
            status = kFail;
        }
        else
        {
            status = kPass;
        }

        nlohmann::json measured;
        measured["max_camera_relative_roundtrip"] = scan.maximum;
        measured["camera_relative_pairs"] = scan.count;
        measured["calib_report_pass"] = reportPass;

        nlohmann::json threshold;
        threshold["roundtrip_atol"] = kThresholds.roundtripAtol;
        threshold["scope"] = "camera-relative T_Ck_rect_V only";

        checks.push_back(MakeCheck("transform_roundtrip", status, std::move(measured),
                                   std::move(threshold),
                                   "world frames are checked rotation-only by "
                                   "rotation_orthonormal and cam0_world"));
    }

    // rotation_orthonormal
    {
        const MetricScan scan =
            ScanCameraKeys(calibReport, "rotation_orth", "rotation_det");
        const bool within = scan.present &&
                            scan.maximum <= std::max(kThresholds.rotationAtol,
                                                     kThresholds.rotationDetAtol);
        const std::string status = StatusFor(within, scan.present && scan.complete);

        nlohmann::json measured;
        measured["max_rotation_orthonormality"] = scan.maximum;
        measured["pairs"] = scan.count;

        nlohmann::json threshold;
        threshold["rotation_atol"] = kThresholds.rotationAtol;
        threshold["rotation_det_atol"] = kThresholds.rotationDetAtol;

        checks.push_back(MakeCheck("rotation_orthonormal", status, std::move(measured),
                                   std::move(threshold),
                                   "every composed rotation must be orthonormal with det +1"));
    }

    // cam0_world
    {
        bool present = false;
        double delta = 0.0;
        if (calibReport.is_object())
        {
            const auto found = calibReport.find("max_cam0_world_delta");
            if (found != calibReport.end())
            {
                present = TryGetNumber(*found, delta);
            }
        }
        const bool within = present && delta <= kThresholds.cam0WorldAtol;
        const std::string status = StatusFor(within, present);

        nlohmann::json measured;
        measured["max_cam0_world_delta"] = present ? nlohmann::json(delta)
                                                   : nlohmann::json(nullptr);

        nlohmann::json threshold;
        threshold["cam0_world_atol"] = kThresholds.cam0WorldAtol;

        checks.push_back(MakeCheck("cam0_world", status, std::move(measured),
                                   std::move(threshold),
                                   "T_W_C0_rect compared against cam0_to_world.txt"));
    }

    // camera_height
    {
        const std::string heightStatus = GetString(calibReport, "camera_height_status");
        std::vector<double> heights;
        bool present = false;
        bool complete = true;
        if (calibReport.is_object())
        {
            const auto found = calibReport.find("camera_height_m");
            if (found != calibReport.end() && found->is_array() && !found->empty())
            {
                present = true;
                for (const nlohmann::json &value : *found)
                {
                    double height = 0.0;
                    if (!TryGetNumber(value, height))
                    {
                        complete = false;
                        continue;
                    }
                    heights.push_back(height);
                }
            }
        }

        const bool skipped = heightStatus == "skipped_no_lidar";
        bool inBand = !heights.empty();
        double spread = 0.0;
        if (!heights.empty())
        {
            const auto [minimum, maximum] =
                std::minmax_element(heights.begin(), heights.end());
            spread = *maximum - *minimum;
            inBand = std::all_of(heights.begin(), heights.end(), [](double height) {
                return height >= kThresholds.cameraHeightMinM &&
                       height <= kThresholds.cameraHeightMaxM;
            });
        }
        const bool stable = spread <= kThresholds.cameraHeightStabilityM;
        const bool passed = present && complete && !heights.empty() && !skipped &&
                            inBand && stable && heightStatus != "failed";

        std::string status = kPass;
        if (!present || !complete || heights.empty())
        {
            status = kMissing;
        }
        else if (!passed)
        {
            status = kFail;
        }

        nlohmann::json measured;
        measured["heights_m"] = heights;
        measured["spread_m"] = spread;
        measured["status"] = heightStatus;
        measured["skipped_no_lidar"] = skipped;

        nlohmann::json threshold;
        threshold["min_m"] = kThresholds.cameraHeightMinM;
        threshold["max_m"] = kThresholds.cameraHeightMaxM;
        threshold["max_spread_m"] = kThresholds.cameraHeightStabilityM;
        threshold["forbidden_status"] = "skipped_no_lidar";

        checks.push_back(MakeCheck("camera_height", status, std::move(measured),
                                   std::move(threshold),
                                   "robust local ground plane, perpendicular distance"));
    }

    // overlay_frames
    {
        std::set<int> frames;
        bool framesPresent = false;
        if (overlayReport.is_object())
        {
            const auto found = overlayReport.find("frames");
            if (found != overlayReport.end() && found->is_array())
            {
                framesPresent = true;
                for (const nlohmann::json &value : *found)
                {
                    if (value.is_number_integer())
                    {
                        frames.insert(value.get<int>());
                    }
                }
            }
        }
        std::set<std::string> cams;
        bool camsPresent = false;
        if (overlayReport.is_object())
        {
            const auto found = overlayReport.find("cams");
            if (found != overlayReport.end() && found->is_array())
            {
                camsPresent = true;
                for (const nlohmann::json &value : *found)
                {
                    if (value.is_string())
                    {
                        cams.insert(value.get<std::string>());
                    }
                }
            }
        }
        const bool bothCameras = cams.count("image_00") != 0U &&
                                 cams.count("image_01") != 0U;
        const bool enoughFrames =
            static_cast<int>(frames.size()) >= kThresholds.overlayMinFrames;
        const bool passed = framesPresent && camsPresent && enoughFrames && bothCameras;
        const std::string status =
            (!framesPresent || !camsPresent) ? kMissing : (passed ? kPass : kFail);

        nlohmann::json measured;
        measured["unique_frames"] = static_cast<int>(frames.size());
        measured["cams"] = cams;

        nlohmann::json threshold;
        threshold["min_frames"] = kThresholds.overlayMinFrames;
        threshold["required_cameras"] = {"image_00", "image_01"};

        checks.push_back(MakeCheck("overlay_frames", status, std::move(measured),
                                   std::move(threshold),
                                   "at least three unique frames and both cameras"));
    }

    // rectified_data_rect_only
    {
        const bool flag = GetBool(overlayReport, kRectifiedFlag, false);
        bool resultsPresent = false;
        bool allOk = true;
        std::vector<std::string> bad;
        if (overlayReport.is_object())
        {
            const auto found = overlayReport.find("results");
            if (found != overlayReport.end() && found->is_array() && !found->empty())
            {
                resultsPresent = true;
                for (const nlohmann::json &result : *found)
                {
                    const bool rectified = GetBool(result, "rectified", false);
                    const std::string image = GetString(result, "image");
                    const bool inDataRect = HasPathComponent(image, "data_rect");
                    if (!rectified || !inDataRect)
                    {
                        allOk = false;
                        const std::string label =
                            std::to_string(GetInt(result, "frame", 0)) + "/" +
                            GetString(result, "camera");
                        bad.push_back(label);
                    }
                }
            }
        }
        const bool passed = flag && resultsPresent && allOk;
        const std::string status =
            !resultsPresent ? kMissing : (passed ? kPass : kFail);

        nlohmann::json measured;
        measured["rectified_data_rect_only"] = flag;
        measured["results"] = resultsPresent;
        measured["bad_results"] = bad;

        nlohmann::json threshold;
        threshold["rectified_data_rect_only"] = true;
        threshold["every_result_rectified"] = true;

        checks.push_back(MakeCheck("rectified_data_rect_only", status,
                                   std::move(measured), std::move(threshold),
                                   "raw/rect mixing is forbidden"));
    }

    // overlay_offsets
    {
        std::vector<std::pair<int, std::string>> expected;
        bool expectedOk = true;
        std::string expectedError;
        try
        {
            expected = ExpectedPairsFromOverlay(overlayReport);
        }
        catch (const std::exception &error)
        {
            expectedOk = false;
            expectedError = error.what();
        }

        nlohmann::json measured;
        measured["pairs_expected"] = static_cast<int>(expected.size());
        measured["pairs_measured"] = 0;
        measured["max_offset_px"] = nullptr;
        measured["pairs"] = nlohmann::json::array();

        nlohmann::json threshold;
        threshold["max_offset_px"] = kThresholds.overlayMaxOffsetPx;
        threshold["comparison"] = "strictly less than max_offset_px";
        threshold["min_features"] = kThresholds.overlayMinFeatures;

        std::string status;
        nlohmann::json evidence;
        if (verification == nullptr)
        {
            status = kPendingHuman;
            evidence = "awaiting human overlay measurement; no verification file";
        }
        else
        {
            std::vector<std::string> errors;
            if (!expectedOk)
            {
                errors.push_back(expectedError);
            }
            const std::vector<std::string> schemaErrors =
                ValidateOverlayVerification(*verification,
                                            expectedOk ? &expected : nullptr);
            errors.insert(errors.end(), schemaErrors.begin(), schemaErrors.end());

            const bool attestationMode = HasAllBelowAttestation(*verification);
            if (attestationMode)
            {
                measured["attestation"] = kAttestationAllBelow;
                measured["threshold_px"] = kThresholds.overlayMaxOffsetPx;
                threshold["attestation"] = kAttestationAllBelow;
            }

            nlohmann::json pairEntries = nlohmann::json::array();
            int measuredPairs = 0;
            bool hasOffset = false;
            double maxOffset = 0.0;
            if (verification->is_object())
            {
                const auto frames = verification->find("frames");
                if (frames != verification->end() && frames->is_array())
                {
                    for (const nlohmann::json &entry : *frames)
                    {
                        if (!entry.is_object())
                        {
                            continue;
                        }
                        const auto frameIt = entry.find("frame");
                        const auto cameraIt = entry.find("camera");
                        const auto offsetIt = entry.find("max_offset_px");
                        if (frameIt == entry.end() || !frameIt->is_number_integer() ||
                            cameraIt == entry.end() || !cameraIt->is_string())
                        {
                            continue;
                        }
                        const int frame = frameIt->get<int>();
                        const std::string camera = cameraIt->get<std::string>();
                        double offset = 0.0;
                        const bool numericOffset =
                            offsetIt != entry.end() && TryGetNumber(*offsetIt, offset);
                        const bool attestedPair =
                            attestationMode &&
                            (offsetIt == entry.end() || offsetIt->is_null());
                        if (!numericOffset && !attestedPair)
                        {
                            continue;
                        }
                        ++measuredPairs;
                        if (numericOffset)
                        {
                            hasOffset = true;
                            maxOffset = std::max(maxOffset, offset);
                        }
                        nlohmann::json pair = {
                            {"frame", frame},
                            {"camera", camera},
                            {"max_offset_px",
                             numericOffset ? nlohmann::json(offset)
                                           : nlohmann::json(nullptr)},
                            {"features_measured",
                             GetInt(entry, "features_measured", 0)}};
                        if (attestedPair)
                        {
                            pair["attested"] = true;
                        }
                        pairEntries.push_back(std::move(pair));
                        if (numericOffset &&
                            offset >= kThresholds.overlayMaxOffsetPx)
                        {
                            errors.push_back("offset " + FormatValue(offset) +
                                             " px for pair " +
                                             PairLabel(frame, camera) +
                                             " is not < " +
                                             FormatValue(kThresholds.overlayMaxOffsetPx) +
                                             " px");
                        }
                    }
                }
            }

            measured["pairs_measured"] = measuredPairs;
            measured["max_offset_px"] =
                hasOffset ? nlohmann::json(maxOffset) : nlohmann::json(nullptr);
            measured["pairs"] = std::move(pairEntries);
            status = errors.empty() ? kPass : kFail;
            evidence = errors;
        }

        checks.push_back(MakeCheck("overlay_offsets", status, std::move(measured),
                                   std::move(threshold), std::move(evidence)));
    }

    // Aggregate failures.
    nlohmann::json blockingFailures = nlohmann::json::array();
    bool allPass = true;
    for (const nlohmann::json &check : checks)
    {
        const std::string status = GetString(check, "status");
        if (status != kPass)
        {
            allPass = false;
        }
        if (status == kFail || status == kMissing)
        {
            blockingFailures.push_back(GetString(check, "id"));
            const auto evidence = check.find("evidence");
            if (GetString(check, "id") == "overlay_offsets" &&
                evidence != check.end() && evidence->is_array())
            {
                for (const nlohmann::json &item : *evidence)
                {
                    if (item.is_string())
                    {
                        blockingFailures.push_back(item.get<std::string>());
                    }
                }
            }
        }
    }

    std::string sequence = GetString(readiness, "sequence");
    if (sequence.empty())
    {
        sequence = GetString(calibReport, "sequence");
    }
    if (sequence.empty())
    {
        sequence = GetString(overlayReport, "sequence");
    }

    nlohmann::json report;
    report["schema_version"] = 1;
    report["generated_at_utc"] = UtcNowIso8601();
    report["sequence"] = sequence;
    report["checks"] = std::move(checks);
    report["blocking_failures"] = std::move(blockingFailures);
    report["pass"] = allPass;
    return report;
}

std::string RenderMarkdown(const nlohmann::json &report)
{
    std::ostringstream out;
    out << "# G1 Evidence Report\n\n";
    out << "**Sequence:** " << GetString(report, "sequence") << "\n\n";
    out << "**Generated (UTC):** " << GetString(report, "generated_at_utc") << "\n\n";
    out << "Gate statement: G1 passes only when every check below is `pass`. A missing "
           "human overlay measurement is `pending_human`, never `pass`.\n\n";

    out << "| Check | Status | Measured | Threshold | Evidence |\n";
    out << "|---|---|---|---|---|\n";
    if (report.is_object() && report.contains("checks") && report["checks"].is_array())
    {
        for (const nlohmann::json &check : report["checks"])
        {
            const auto foundMeasured = check.find("measured");
            const auto foundThreshold = check.find("threshold");
            const auto foundEvidence = check.find("evidence");
            out << "| " << GetString(check, "id") << " | "
                << GetString(check, "status") << " | "
                << FormatValue(foundMeasured == check.end() ? nlohmann::json()
                                                            : foundMeasured.value())
                << " | "
                << FormatValue(foundThreshold == check.end() ? nlohmann::json()
                                                             : foundThreshold.value())
                << " | "
                << FormatEvidence(foundEvidence == check.end() ? nlohmann::json()
                                                               : foundEvidence.value())
                << " |\n";
        }
    }

    out << "\n## Sampled frame/camera offsets\n\n";
    const nlohmann::json *offsets = FindCheck(report, "overlay_offsets");
    bool wroteAny = false;
    if (offsets != nullptr)
    {
        const auto measured = offsets->find("measured");
        if (measured != offsets->end() && measured->is_object())
        {
            const bool attested =
                measured->value("attestation", std::string()) == kAttestationAllBelow;
            if (attested)
            {
                out << "Reviewer attestation: all measured offsets are < "
                    << FormatValue(kThresholds.overlayMaxOffsetPx)
                    << " px (original 1×).\n\n";
            }
            const auto pairs = measured->find("pairs");
            if (pairs != measured->end() && pairs->is_array() && !pairs->empty())
            {
                out << "| Frame | Camera | max_offset_px (original 1×) | Features |\n";
                out << "|---|---|---|---|\n";
                for (const nlohmann::json &entry : *pairs)
                {
                    out << "| " << GetInt(entry, "frame", 0) << " | "
                        << GetString(entry, "camera") << " | ";
                    const auto offset = entry.find("max_offset_px");
                    if (offset == entry.end() || offset->is_null())
                    {
                        out << (attested ? "attested < 2.0" : "null");
                    }
                    else
                    {
                        out << FormatValue(offset.value());
                    }
                    out << " | " << GetInt(entry, "features_measured", 0) << " |\n";
                    wroteAny = true;
                }
            }
        }
    }
    if (!wroteAny)
    {
        out << "No human-measured offsets are recorded yet.\n";
    }

    out << "\n## Pair coverage\n\n";
    int expectedPairs = 0;
    int measuredPairs = 0;
    if (offsets != nullptr)
    {
        const auto measured = offsets->find("measured");
        if (measured != offsets->end() && measured->is_object())
        {
            expectedPairs = measured->value("pairs_expected", 0);
            measuredPairs = measured->value("pairs_measured", 0);
        }
    }
    out << "- Expected pairs: " << expectedPairs << "\n";
    out << "- Measured pairs: " << measuredPairs << "\n\n";

    out << (GetBool(report, "pass", false) ? "G1 PASSED\n" : "G1 NOT PASSED\n");
    return out.str();
}
} // namespace raylidar::g1
