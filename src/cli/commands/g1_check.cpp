#include <raylidar/cli/command.h>
#include <raylidar/cli/commands.h>
#include <raylidar/common/artifacts.h>
#include <raylidar/common/cli_options.h>
#include <raylidar/common/io.h>
#include <raylidar/g1/g1.h>
#include <raylidar/prereg/prereg.h>

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace
{
namespace fs = std::filesystem;
using raylidar::IoError;
using raylidar::ParsedArgs;
using raylidar::ParseArgs;
using raylidar::PhaseArtifact;
using raylidar::UsageError;
using raylidar::g1::EvaluateG1;
using raylidar::g1::ExpectedPairsFromOverlay;
using raylidar::g1::RenderMarkdown;
using raylidar::g1::Thresholds;
using raylidar::g1::ValidateOverlayVerification;
using raylidar::prereg::VerifyDecisions;

constexpr int kExitAssertionFailure = 5;
constexpr int kExitInvalidInput = 4;
constexpr int kExitPendingHuman = 7;

void PrintHelp(std::ostream &out)
{
    out << "Usage: raylidar_cli g1-check [options]\n"
        << "\n"
        << "Aggregate real-sequence G1 evidence and emit the measurement worksheet.\n"
        << "A missing human overlay measurement is reported as pending_human.\n"
        << "\n"
        << "Options:\n"
        << "  --config <path>                 Frames/config YAML (default: configs/frames.yaml)\n"
        << "  --calib-report <path>           calib_report.json (default: <artifacts>/01/calib/)\n"
        << "  --overlay-report <path>         overlay_report.json (default: <artifacts>/01/overlays/)\n"
        << "  --readiness <path>              readiness.json (default: <artifacts>/01/readiness.json)\n"
        << "  --preflight <path>              preflight_after.json (default: <artifacts>/01/)\n"
        << "  --prereg-json <path>            decisions.sha256.json (default: <artifacts>/01/)\n"
        << "  --verification <path>           overlay_verification.json (optional)\n"
        << "  --out <path>                    Report base path (default: <artifacts>/01/G1_REPORT)\n"
        << "  --emit-worksheet <path>         Write the human measurement worksheet\n"
        << "  --validate-verification <path>  Validate a verification file and exit\n"
        << "  --decisions <path>              Decisions document (default: docs/decisions.md)\n"
        << "  --sidecar <path>                Hash sidecar (default: <artifacts>/01/decisions.sha256)\n"
        << "  --help                          Show this help\n"
        << "\n"
        << "Exit codes: 0 pass, 4 missing/invalid input, 5 evidence failure,\n"
        << "7 overlay_offsets pending human measurement.\n";
}

double NodeDouble(const YAML::Node &node, const char *key, double fallback)
{
    if (!node || !node.IsMap())
    {
        return fallback;
    }
    const YAML::Node value = node[key];
    if (!value || !value.IsScalar())
    {
        return fallback;
    }
    try
    {
        return value.as<double>();
    }
    catch (const YAML::Exception &)
    {
        return fallback;
    }
}

int NodeInt(const YAML::Node &node, const char *key, int fallback)
{
    if (!node || !node.IsMap())
    {
        return fallback;
    }
    const YAML::Node value = node[key];
    if (!value || !value.IsScalar())
    {
        return fallback;
    }
    try
    {
        return value.as<int>();
    }
    catch (const YAML::Exception &)
    {
        return fallback;
    }
}

std::pair<double, double> NodePair(const YAML::Node &node, const char *key, double firstFallback,
                                   double secondFallback)
{
    if (!node || !node.IsMap())
    {
        return {firstFallback, secondFallback};
    }
    const YAML::Node value = node[key];
    if (!value || !value.IsSequence() || value.size() != 2U)
    {
        return {firstFallback, secondFallback};
    }
    try
    {
        return {value[0].as<double>(), value[1].as<double>()};
    }
    catch (const YAML::Exception &)
    {
        return {firstFallback, secondFallback};
    }
}

bool CloseEnough(double actual, double expected)
{
    return std::abs(actual - expected) <=
           1e-15 * std::max(1.0, std::abs(expected));
}

/// The CLI must never take thresholds from YAML: thresholds are code constants.
/// Fail loudly if a config drifts away from `G1Thresholds`.
void CheckConfigMatchesThresholds(const YAML::Node &config, const fs::path &path)
{
    const auto &thresholds = Thresholds();
    const auto mismatch = [&path](const std::string &key) {
        throw IoError("config " + path.string() + " " + key +
                          " disagrees with the code-constant G1Thresholds",
                      kExitInvalidInput);
    };

    const YAML::Node assertions = config["assertions"];
    if (!CloseEnough(NodeDouble(assertions, "roundtrip_atol", thresholds.roundtripAtol),
                     thresholds.roundtripAtol))
    {
        mismatch("assertions.roundtrip_atol");
    }
    if (!CloseEnough(NodeDouble(assertions, "rotation_atol", thresholds.rotationAtol),
                     thresholds.rotationAtol))
    {
        mismatch("assertions.rotation_atol");
    }
    if (!CloseEnough(NodeDouble(assertions, "rotation_det_atol",
                                thresholds.rotationDetAtol),
                     thresholds.rotationDetAtol))
    {
        mismatch("assertions.rotation_det_atol");
    }
    if (!CloseEnough(NodeDouble(assertions, "cam0_world_atol", thresholds.cam0WorldAtol),
                     thresholds.cam0WorldAtol))
    {
        mismatch("assertions.cam0_world_atol");
    }
    const auto heights = NodePair(assertions, "camera_height_m", thresholds.cameraHeightMinM,
                                  thresholds.cameraHeightMaxM);
    if (!CloseEnough(heights.first, thresholds.cameraHeightMinM) ||
        !CloseEnough(heights.second, thresholds.cameraHeightMaxM))
    {
        mismatch("assertions.camera_height_m");
    }
    if (!CloseEnough(NodeDouble(assertions, "camera_height_stability_m",
                                thresholds.cameraHeightStabilityM),
                     thresholds.cameraHeightStabilityM))
    {
        mismatch("assertions.camera_height_stability_m");
    }

    const YAML::Node overlay = config["overlay"];
    if (NodeInt(overlay, "min_frames", thresholds.overlayMinFrames) !=
        thresholds.overlayMinFrames)
    {
        mismatch("overlay.min_frames");
    }
    if (NodeInt(overlay, "min_features", thresholds.overlayMinFeatures) !=
        thresholds.overlayMinFeatures)
    {
        mismatch("overlay.min_features");
    }
    if (!CloseEnough(NodeDouble(overlay, "max_offset_px", thresholds.overlayMaxOffsetPx),
                     thresholds.overlayMaxOffsetPx))
    {
        mismatch("overlay.max_offset_px");
    }
}

std::string JsonString(const nlohmann::json &object, const char *key)
{
    if (!object.is_object())
    {
        return {};
    }
    const auto found = object.find(key);
    if (found == object.end() || !found->is_string())
    {
        return {};
    }
    return found->get<std::string>();
}

int JsonInt(const nlohmann::json &object, const char *key, int fallback)
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

void WriteText(const fs::path &path, const std::string &text)
{
    if (path.has_parent_path())
    {
        raylidar::EnsureDir(path.parent_path());
    }
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output)
    {
        throw IoError("cannot write " + path.string(), kExitInvalidInput);
    }
    output.write(text.data(), static_cast<std::streamsize>(text.size()));
    if (!output)
    {
        throw IoError("failed to write " + path.string(), kExitInvalidInput);
    }
}

void RequireFile(const fs::path &path)
{
    std::error_code error;
    if (!fs::is_regular_file(path, error))
    {
        throw IoError("missing input: " + path.string(), kExitInvalidInput);
    }
}

/// Every verification entry must use original 1x pixels and stay strictly below
/// the code-constant gate.
void AppendOffsetErrors(const nlohmann::json &verification,
                        std::vector<std::string> &errors)
{
    if (!verification.is_object())
    {
        return;
    }
    const auto frames = verification.find("frames");
    if (frames == verification.end() || !frames->is_array())
    {
        return;
    }
    for (const nlohmann::json &entry : *frames)
    {
        if (!entry.is_object())
        {
            continue;
        }
        const int frame = JsonInt(entry, "frame", 0);
        const std::string camera = JsonString(entry, "camera");
        const auto offset = entry.find("max_offset_px");
        if (offset == entry.end() || !offset->is_number())
        {
            continue;
        }
        const double pixels = offset->get<double>();
        if (pixels >= Thresholds().overlayMaxOffsetPx)
        {
            errors.push_back("offset " + std::to_string(pixels) + " px for pair (" +
                             std::to_string(frame) + ", " + camera +
                             ") is not < " +
                             std::to_string(Thresholds().overlayMaxOffsetPx) + " px");
        }
    }
}

void WriteWorksheet(const fs::path &path, const nlohmann::json &overlayReport)
{
    const auto expected = ExpectedPairsFromOverlay(overlayReport);

    std::map<std::pair<int, std::string>, nlohmann::json> results;
    const auto resultsIt = overlayReport.find("results");
    if (resultsIt != overlayReport.end() && resultsIt->is_array())
    {
        for (const nlohmann::json &result : *resultsIt)
        {
            if (result.is_object() && result.contains("frame") &&
                result["frame"].is_number_integer())
            {
                results[{result["frame"].get<int>(), JsonString(result, "camera")}] =
                    result;
            }
        }
    }

    std::ostringstream out;
    out << "# G1 Overlay Measurement Worksheet\n\n";
    out << "Every pair below projects LiDAR into a rectified `data_rect` image. "
           "Measure at least five features per pair on the 4× PNG.\n\n";
    out << "**Unit:** `max_offset_px` is in original 1× pixels; measure on the 4× PNG "
           "and divide by `scale_factor`.\n\n";
    out << "| Frame | Camera | Image | Image resolved | Overlay 4× | scale_factor |\n";
    out << "|---|---|---|---|---|---|\n";
    for (const auto &pair : expected)
    {
        const auto found = results.find(pair);
        if (found == results.end())
        {
            continue;
        }
        const nlohmann::json &result = found->second;
        out << "| " << pair.first << " | " << pair.second << " | "
            << JsonString(result, "image") << " | "
            << JsonString(result, "image_resolved") << " | "
            << JsonString(result, "overlay_4x") << " | "
            << JsonInt(result, "scale_factor", 4) << " |\n";
    }

    for (const auto &pair : expected)
    {
        const auto found = results.find(pair);
        if (found == results.end())
        {
            continue;
        }
        const nlohmann::json &result = found->second;
        const int scale = JsonInt(result, "scale_factor", 4);
        out << "\n## Frame " << pair.first << " / " << pair.second << "\n\n";
        out << "- `image`: " << JsonString(result, "image") << "\n";
        out << "- `image_resolved`: " << JsonString(result, "image_resolved") << "\n";
        out << "- `overlay_4x`: " << JsonString(result, "overlay_4x") << "\n";
        out << "- `scale_factor`: " << scale << "\n\n";
        out << "**Unit:** `max_offset_px` is in original 1× pixels; measure on the "
               "4× PNG and divide by `scale_factor`.\n\n";
        out << "| feature | measured_offset_4x_px | max_offset_px | notes |\n";
        out << "|---|---|---|---|\n";
        for (int feature = 1; feature <= Thresholds().overlayMinFeatures; ++feature)
        {
            out << "| feature_" << feature << " |  |  |  |\n";
        }
    }

    WriteText(path, out.str());
}

int Execute(std::span<const std::string_view> args)
{
    const ParsedArgs parsed =
        ParseArgs(args,
                  {"config", "calib-report", "overlay-report", "readiness", "preflight",
                   "prereg-json", "verification", "out", "emit-worksheet",
                   "validate-verification", "decisions", "sidecar"},
                  {"help"});
    if (parsed.Flag("help"))
    {
        PrintHelp(std::cout);
        return 0;
    }

    const fs::path configPath = parsed.Get("config", "configs/frames.yaml");
    const YAML::Node config = raylidar::ReadYaml(configPath);
    if (!config || !config.IsMap())
    {
        throw IoError(configPath.string() + ": expected a YAML mapping", kExitInvalidInput);
    }
    CheckConfigMatchesThresholds(config, configPath);

    const fs::path overlayReportPath = parsed.Get(
        "overlay-report", PhaseArtifact(1, "overlays/overlay_report.json").string());

    if (parsed.Has("validate-verification"))
    {
        RequireFile(overlayReportPath);
        const nlohmann::json overlayReport = raylidar::ReadJson(overlayReportPath);
        const auto expected = ExpectedPairsFromOverlay(overlayReport);

        const fs::path verificationPath = parsed.Get("validate-verification");
        RequireFile(verificationPath);
        const nlohmann::json verification = raylidar::ReadJson(verificationPath);

        std::vector<std::string> errors =
            ValidateOverlayVerification(verification, &expected);
        AppendOffsetErrors(verification, errors);
        if (errors.empty())
        {
            std::cout << "verification OK: " << expected.size() << " pair(s)\n";
            return 0;
        }
        for (const std::string &error : errors)
        {
            std::cerr << "error: " << error << '\n';
        }
        return kExitAssertionFailure;
    }

    const fs::path calibReportPath = parsed.Get(
        "calib-report", PhaseArtifact(1, "calib/calib_report.json").string());
    const fs::path readinessPath =
        parsed.Get("readiness", PhaseArtifact(1, "readiness.json").string());
    const fs::path preflightPath =
        parsed.Get("preflight", PhaseArtifact(1, "preflight_after.json").string());
    const fs::path preregPath = parsed.Get(
        "prereg-json", PhaseArtifact(1, "decisions.sha256.json").string());

    RequireFile(calibReportPath);
    RequireFile(overlayReportPath);
    RequireFile(readinessPath);
    RequireFile(preflightPath);
    RequireFile(preregPath);

    nlohmann::json prereg = raylidar::ReadJson(preregPath);
    const fs::path decisionsPath = parsed.Get("decisions", "docs/decisions.md");
    const fs::path sidecarPath =
        parsed.Get("sidecar", PhaseArtifact(1, "decisions.sha256").string());
    const nlohmann::json verified = VerifyDecisions(decisionsPath, sidecarPath);
    prereg["ok"] = verified["ok"];
    prereg["expected"] = verified["expected"];
    prereg["actual"] = verified["actual"];
    prereg["frozen_at_utc"] = verified["frozen_at_utc"];

    const fs::path verificationPath = parsed.Get(
        "verification", PhaseArtifact(1, "overlay_verification.json").string());
    nlohmann::json verification;
    bool haveVerification = false;
    if (parsed.Has("verification"))
    {
        RequireFile(verificationPath);
        verification = raylidar::ReadJson(verificationPath);
        haveVerification = true;
    }
    else
    {
        std::error_code error;
        if (fs::is_regular_file(verificationPath, error))
        {
            verification = raylidar::ReadJson(verificationPath);
            haveVerification = true;
        }
    }

    const nlohmann::json preflight = raylidar::ReadJson(preflightPath);
    const nlohmann::json readiness = raylidar::ReadJson(readinessPath);
    const nlohmann::json calibReport = raylidar::ReadJson(calibReportPath);
    const nlohmann::json overlayReport = raylidar::ReadJson(overlayReportPath);

    const nlohmann::json report =
        EvaluateG1(preflight, readiness, calibReport, overlayReport, prereg,
                   haveVerification ? &verification : nullptr);

    if (parsed.Has("emit-worksheet"))
    {
        const fs::path worksheetPath = parsed.Get("emit-worksheet");
        if (worksheetPath.empty())
        {
            throw IoError("--emit-worksheet requires a path", kExitInvalidInput);
        }
        WriteWorksheet(worksheetPath, overlayReport);
    }

    const fs::path outBase =
        parsed.Get("out", PhaseArtifact(1, "G1_REPORT").string());
    raylidar::WriteJson(outBase.string() + ".json", report);
    const std::string markdown = RenderMarkdown(report);
    WriteText(outBase.string() + ".md", markdown);
    std::cout << markdown;

    if (report.value("pass", false))
    {
        return 0;
    }

    bool pending = false;
    bool failure = false;
    for (const nlohmann::json &check : report["checks"])
    {
        const std::string status = JsonString(check, "status");
        if (JsonString(check, "id") == "overlay_offsets" && status == "pending_human")
        {
            pending = true;
        }
        else if (status == "fail" || status == "missing")
        {
            failure = true;
        }
    }
    if (failure)
    {
        return kExitAssertionFailure;
    }
    if (pending)
    {
        return kExitPendingHuman;
    }
    return kExitAssertionFailure;
}
} // namespace

int RunG1Check(std::span<const std::string_view> args)
{
    try
    {
        return Execute(args);
    }
    catch (const UsageError &error)
    {
        std::cerr << "error: " << error.what() << "\n\n";
        PrintHelp(std::cerr);
        return error.exitCode();
    }
    catch (const IoError &error)
    {
        std::cerr << "error: " << error.what() << '\n';
        return error.exitCode();
    }
    catch (const std::out_of_range &error)
    {
        std::cerr << "error: " << error.what() << '\n';
        return kExitInvalidInput;
    }
    catch (const std::invalid_argument &error)
    {
        std::cerr << "error: " << error.what() << '\n';
        return kExitAssertionFailure;
    }
    catch (const std::runtime_error &error)
    {
        std::cerr << "error: " << error.what() << '\n';
        return kExitAssertionFailure;
    }
    catch (const std::exception &error)
    {
        std::cerr << "error: " << error.what() << '\n';
        return kExitInvalidInput;
    }
}

RAYLIDAR_REGISTER_COMMAND(
    "g1-check", "Aggregate G1 evidence and emit the measurement worksheet/report",
    RunG1Check);
