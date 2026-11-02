#include <raylidar/calib/kitti360.h>
#include <raylidar/cli/command.h>
#include <raylidar/cli/commands_phase2.h>
#include <raylidar/common/artifacts.h>
#include <raylidar/common/cli_options.h>
#include <raylidar/common/io.h>
#include <raylidar/common/mask_bits.h>
#include <raylidar/lidar/uncurl_diag.h>

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace
{
namespace fs = std::filesystem;
using raylidar::IoError;
using raylidar::ParseArgs;
using raylidar::ParsedArgs;
using raylidar::PhaseArtifactsDir;
using raylidar::UsageError;
using raylidar::UtcNowIso8601;
using raylidar::WriteJson;
using raylidar::calib::Kitti360Calib;
using raylidar::calib::LoadKitti360Calib;
using raylidar::calib::LoadPoses;
using raylidar::lidar::AzimuthBin;
using raylidar::lidar::EvaluateUncurl;
using raylidar::lidar::UncurlDiagnostic;
using raylidar::lidar::UncurlThresholds;

constexpr int kExitInvalidInput = 4;
constexpr int kExitGateFailure = 6;

void PrintHelp(std::ostream &out)
{
    out << "Usage: raylidar_cli uncurl-check [options]\n"
        << "\n"
        << "Run the masked raw-vs-curled azimuth-flatness diagnostic. Every point\n"
        << "must have a per-point mask bit; run `mask-build` first. There is no\n"
        << "unmasked fallback for real frames.\n"
        << "\n"
        << "Options:\n"
        << "  --config <path>       Phase-2 YAML (default: configs/phase2.yaml)\n"
        << "  --frames <list>       Comma-separated ascending frame ids\n"
        << "  --frame-list <path>   JSON with usable_frames or selected.frames\n"
        << "                        (default: <phase_dir>/segment/segment_manifest.json)\n"
        << "  --mask-dir <path>     Per-frame mask directory (default: <phase_dir>/masks)\n"
        << "  --max-frames <n>      Deterministic smoke cap (default: all frames)\n"
        << "  --out-dir <path>      Artifact directory (default: <phase_dir>/uncurl)\n"
        << "  --help                Show this help\n"
        << "\n"
        << "Prerequisite: `raylidar_cli mask-build` must have written\n"
        << "masks/frame_<frame>/points.bin for every requested frame.\n"
        << "\n"
        << "Exit codes: 0 gate passed, 2 usage error, 4 invalid input/missing mask,\n"
        << "6 flatness gate failed.\n";
}

std::string NodeString(const YAML::Node &node, const char *key, const std::string &fallback)
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
    return value.as<std::string>();
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

bool NodeBool(const YAML::Node &node, const char *key, bool fallback)
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
        return value.as<bool>();
    }
    catch (const YAML::Exception &)
    {
        return fallback;
    }
}

void Require(const bool condition, const std::string &message)
{
    if (!condition)
    {
        throw IoError(message, kExitInvalidInput);
    }
}

std::string Trim(const std::string &value)
{
    const auto notSpace = [](const unsigned char ch) { return std::isspace(ch) == 0; };
    auto begin = std::find_if(value.begin(), value.end(), notSpace);
    auto end = std::find_if(value.rbegin(), value.rend(), notSpace).base();
    if (begin >= end)
    {
        return {};
    }
    return std::string(begin, end);
}

std::vector<int> ParseFrameList(const std::string &text, const std::string &source)
{
    std::vector<int> frames;
    std::size_t start = 0;
    while (start <= text.size())
    {
        const std::size_t comma = text.find(',', start);
        const std::string token =
            Trim(text.substr(start, comma == std::string::npos ? std::string::npos : comma - start));
        if (!token.empty())
        {
            try
            {
                std::size_t consumed = 0;
                const int frame = std::stoi(token, &consumed);
                if (consumed != token.size())
                {
                    throw std::invalid_argument("trailing characters");
                }
                frames.push_back(frame);
            }
            catch (const std::exception &)
            {
                throw IoError(source + ": invalid frame id '" + token + "'", kExitInvalidInput);
            }
        }
        if (comma == std::string::npos)
        {
            break;
        }
        start = comma + 1U;
    }
    if (frames.empty())
    {
        throw IoError(source + ": no frame ids supplied", kExitInvalidInput);
    }
    return frames;
}

std::vector<int> FramesFromJson(const fs::path &path)
{
    const nlohmann::json document = raylidar::ReadJson(path);
    const nlohmann::json *array = nullptr;
    if (document.contains("usable_frames") && document["usable_frames"].is_array())
    {
        array = &document["usable_frames"];
    }
    else if (document.contains("selected") && document["selected"].is_object() &&
             document["selected"].contains("frames") && document["selected"]["frames"].is_array())
    {
        array = &document["selected"]["frames"];
    }
    if (array == nullptr)
    {
        throw IoError(path.string() + ": expected usable_frames or selected.frames", kExitInvalidInput);
    }

    std::vector<int> frames;
    frames.reserve(array->size());
    for (const nlohmann::json &value : *array)
    {
        if (!value.is_number_integer())
        {
            throw IoError(path.string() + ": frame ids must be integers", kExitInvalidInput);
        }
        frames.push_back(value.get<int>());
    }
    if (frames.empty())
    {
        throw IoError(path.string() + ": frame list is empty", kExitInvalidInput);
    }
    return frames;
}

nlohmann::json BinsJson(const std::vector<AzimuthBin> &bins)
{
    nlohmann::json array = nlohmann::json::array();
    for (const AzimuthBin &bin : bins)
    {
        nlohmann::json entry;
        entry["azimuth_deg"] = bin.centerDeg;
        entry["count"] = bin.count;
        entry["mean_signed_range_error_m"] = bin.meanSignedRangeErrorM;
        array.push_back(std::move(entry));
    }
    return array;
}

nlohmann::json VariantJson(const std::vector<AzimuthBin> &bins, const double peakToPeakM, const double maxAbsBinMeanM,
                           const double sinusoidAmplitudeM, const std::size_t matchedCount)
{
    nlohmann::json variant;
    variant["peak_to_peak_m"] = peakToPeakM;
    variant["max_abs_bin_mean_m"] = maxAbsBinMeanM;
    variant["sinusoid_amplitude_m"] = sinusoidAmplitudeM;
    variant["matched_count"] = matchedCount;
    variant["bins"] = BinsJson(bins);
    return variant;
}

std::string FrameName(const int frame)
{
    char buffer[16] = {};
    std::snprintf(buffer, sizeof(buffer), "%010d", frame);
    return std::string(buffer);
}

/// Independent mask precondition: every requested frame must have a per-point
/// mask whose kept-static bits are read through the frozen `IsKeptStatic`
/// query. A missing or wrong-length mask is exit 4; there is no fallback.
std::size_t ValidateMasks(const std::vector<int> &frames, const fs::path &maskDir, const fs::path &lidarRoot)
{
    std::size_t keptStaticTotal = 0;
    for (const int frame : frames)
    {
        const fs::path scanPath = lidarRoot / "data" / (FrameName(frame) + ".bin");
        std::error_code error;
        const std::uintmax_t scanBytes = std::filesystem::file_size(scanPath, error);
        if (error || scanBytes == 0U || scanBytes % 16U != 0U)
        {
            throw IoError("cannot derive point count for frame " + std::to_string(frame) + ": " + scanPath.string(),
                          kExitInvalidInput);
        }
        const std::size_t pointCount = static_cast<std::size_t>(scanBytes / 16U);

        const fs::path maskPath = maskDir / ("frame_" + FrameName(frame)) / "points.bin";
        const std::uintmax_t maskBytes = std::filesystem::file_size(maskPath, error);
        if (error)
        {
            throw IoError("missing point mask " + maskPath.string() +
                              " (run mask-build before uncurl-check): " + error.message(),
                          kExitInvalidInput);
        }
        if (maskBytes != pointCount)
        {
            throw IoError("point mask " + maskPath.string() + " has " + std::to_string(maskBytes) +
                              " bits but the scan has " + std::to_string(pointCount) + " points",
                          kExitInvalidInput);
        }
        std::ifstream input(maskPath, std::ios::binary);
        if (!input.good())
        {
            throw IoError("cannot read point mask " + maskPath.string(), kExitInvalidInput);
        }
        std::vector<std::uint8_t> bits(pointCount);
        input.read(reinterpret_cast<char *>(bits.data()), static_cast<std::streamsize>(bits.size()));
        if (!input.good())
        {
            throw IoError("truncated point mask " + maskPath.string(), kExitInvalidInput);
        }
        for (const std::uint8_t bit : bits)
        {
            if (raylidar::masking::IsKeptStatic(bit))
            {
                ++keptStaticTotal;
            }
        }
    }
    return keptStaticTotal;
}

void WriteCsv(const fs::path &path, const UncurlDiagnostic &diagnostic)
{
    std::ofstream output(path, std::ios::trunc);
    if (!output.good())
    {
        throw IoError("cannot write " + path.string(), kExitInvalidInput);
    }
    output << "variant,azimuth_deg,count,mean_signed_range_error_m\n";
    output << std::setprecision(12);
    const auto writeVariant = [&output](const char *name, const std::vector<AzimuthBin> &bins) {
        for (const AzimuthBin &bin : bins)
        {
            output << name << ',' << bin.centerDeg << ',' << bin.count << ',' << bin.meanSignedRangeErrorM << '\n';
        }
    };
    writeVariant("raw", diagnostic.rawBins);
    writeVariant("uncurled", diagnostic.uncurledBins);
    if (!output.good())
    {
        throw IoError("error while writing " + path.string(), kExitInvalidInput);
    }
}

int Execute(std::span<const std::string_view> args)
{
    const ParsedArgs parsed =
        ParseArgs(args, {"config", "frames", "frame-list", "mask-dir", "max-frames", "out-dir"}, {"help"});
    if (parsed.Flag("help"))
    {
        PrintHelp(std::cout);
        return 0;
    }

    const std::string configPath = parsed.Get("config", "configs/phase2.yaml");
    const YAML::Node config = raylidar::ReadYaml(configPath);
    Require(config && config.IsMap(), configPath + ": expected a YAML mapping");

    const std::string datasetRoot = NodeString(config, "dataset_root", "");
    const std::string lidarRoot = NodeString(config, "lidar_root", "");
    const std::string sequence = NodeString(config, "sequence", "");
    Require(!datasetRoot.empty(), configPath + ": dataset_root is not configured");
    Require(!lidarRoot.empty(), configPath + ": lidar_root is not configured");
    Require(!sequence.empty(), configPath + ": sequence is not configured");

    fs::path phaseDir = NodeString(config["artifacts"], "phase_dir", "");
    if (phaseDir.empty())
    {
        phaseDir = PhaseArtifactsDir(2);
    }

    const YAML::Node uncurl = config["uncurl"];
    Require(uncurl && uncurl.IsMap(), configPath + ": missing uncurl section");
    Require(NodeString(uncurl, "azimuth_convention", "") == "0.5_plus_atan2_over_2pi",
            configPath + ": uncurl.azimuth_convention must be 0.5_plus_atan2_over_2pi");
    Require(NodeString(uncurl, "interval", "") == "current_to_next_timestamp",
            configPath + ": uncurl.interval must be current_to_next_timestamp");
    Require(NodeInt(uncurl, "max_next_pose_delta_frames", -1) == 0,
            configPath + ": uncurl.max_next_pose_delta_frames must be 0");
    Require(NodeBool(uncurl, "require_ablation_reported", false),
            configPath + ": uncurl.require_ablation_reported must be true");

    UncurlThresholds thresholds;
    thresholds.maxAbsBinMeanM = NodeDouble(uncurl, "max_abs_bin_mean_m", 0.05);
    thresholds.maxSinusoidAmplitudeM = NodeDouble(uncurl, "max_sinusoid_amplitude_m", 0.03);
    thresholds.binWidthDeg = NodeDouble(uncurl, "azimuth_bins_deg", 5.0);
    thresholds.matchVoxelM = NodeDouble(uncurl, "match_voxel_m", 0.2);
    thresholds.maxMatchDistanceM = NodeDouble(uncurl, "max_match_distance_m", 0.3);
    Require(thresholds.maxAbsBinMeanM > 0.0, configPath + ": max_abs_bin_mean_m must be positive");
    Require(thresholds.maxSinusoidAmplitudeM > 0.0, configPath + ": max_sinusoid_amplitude_m must be positive");
    Require(thresholds.binWidthDeg > 0.0, configPath + ": azimuth_bins_deg must be positive");
    Require(thresholds.matchVoxelM > 0.0, configPath + ": match_voxel_m must be positive");
    Require(thresholds.maxMatchDistanceM > 0.0, configPath + ": max_match_distance_m must be positive");

    fs::path maskDir = parsed.Get("mask-dir", "");
    if (maskDir.empty())
    {
        maskDir = phaseDir / "masks";
    }
    fs::path outDir = parsed.Get("out-dir", "");
    if (outDir.empty())
    {
        outDir = phaseDir / "uncurl";
    }

    std::vector<int> frames;
    const std::string framesText = parsed.Get("frames", "");
    if (!framesText.empty())
    {
        frames = ParseFrameList(framesText, "--frames");
    }
    else
    {
        fs::path frameListPath = parsed.Get("frame-list", "");
        if (frameListPath.empty())
        {
            frameListPath = phaseDir / "segment" / "segment_manifest.json";
        }
        frames = FramesFromJson(frameListPath);
    }
    std::sort(frames.begin(), frames.end());
    frames.erase(std::unique(frames.begin(), frames.end()), frames.end());

    const std::string maxFramesText = parsed.Get("max-frames", "");
    if (!maxFramesText.empty())
    {
        int maxFrames = 0;
        try
        {
            std::size_t consumed = 0;
            maxFrames = std::stoi(maxFramesText, &consumed);
            Require(consumed == maxFramesText.size() && maxFrames > 0, "--max-frames must be a positive integer");
        }
        catch (const std::invalid_argument &)
        {
            throw IoError("--max-frames must be a positive integer", kExitInvalidInput);
        }
        if (frames.size() > static_cast<std::size_t>(maxFrames))
        {
            frames.resize(static_cast<std::size_t>(maxFrames));
        }
    }
    Require(frames.size() >= 2U, "uncurl-check needs at least two frames");

    const std::size_t keptStaticTotal = ValidateMasks(frames, maskDir, lidarRoot);

    const Kitti360Calib calib = LoadKitti360Calib(fs::path(datasetRoot) / "calibration");
    const std::map<int, Eigen::Matrix4d> poses =
        LoadPoses(fs::path(datasetRoot) / "data_poses" / sequence / "poses.txt");

    const UncurlDiagnostic diagnostic =
        EvaluateUncurl(frames, datasetRoot, lidarRoot, maskDir, calib, poses, thresholds);

    raylidar::EnsureDir(outDir);
    nlohmann::json document;
    document["schema_version"] = 1;
    document["generated_at_utc"] = UtcNowIso8601();
    document["sequence"] = sequence;
    document["frames"] = frames;
    document["dataset_root"] = datasetRoot;
    document["lidar_root"] = lidarRoot;
    document["mask_dir"] = maskDir.string();
    document["masks_required"] = true;
    document["thresholds"] = {{"max_abs_bin_mean_m", thresholds.maxAbsBinMeanM},
                              {"max_sinusoid_amplitude_m", thresholds.maxSinusoidAmplitudeM},
                              {"azimuth_bins_deg", thresholds.binWidthDeg},
                              {"match_voxel_m", thresholds.matchVoxelM},
                              {"max_match_distance_m", thresholds.maxMatchDistanceM},
                              {"min_bin_samples", thresholds.minBinSamples}};
    document["raw"] = VariantJson(diagnostic.rawBins, diagnostic.rawPeakToPeakM, diagnostic.rawMaxAbsBinMeanM,
                                  diagnostic.rawSinusoidAmplitudeM, diagnostic.rawMatchedCount);
    document["uncurled"] =
        VariantJson(diagnostic.uncurledBins, diagnostic.uncurledPeakToPeakM, diagnostic.uncurledMaxAbsBinMeanM,
                    diagnostic.uncurledSinusoidAmplitudeM, diagnostic.uncurledMatchedCount);
    document["rawPeakToPeakM"] = diagnostic.rawPeakToPeakM;
    document["uncurledPeakToPeakM"] = diagnostic.uncurledPeakToPeakM;
    document["rawMaxAbsBinMeanM"] = diagnostic.rawMaxAbsBinMeanM;
    document["uncurledMaxAbsBinMeanM"] = diagnostic.uncurledMaxAbsBinMeanM;
    document["rawSinusoidAmplitudeM"] = diagnostic.rawSinusoidAmplitudeM;
    document["uncurledSinusoidAmplitudeM"] = diagnostic.uncurledSinusoidAmplitudeM;
    document["pass"] = diagnostic.pass;
    document["failures"] = diagnostic.failures;

    WriteJson(outDir / "uncurl_report.json", document);
    WriteCsv(outDir / "azimuth_bins.csv", diagnostic);

    std::cout << "frames=" << frames.size() << " kept_static_points=" << keptStaticTotal
              << " raw_matched=" << diagnostic.rawMatchedCount
              << " uncurled_matched=" << diagnostic.uncurledMatchedCount
              << " uncurled_max_abs_bin_mean_m=" << diagnostic.uncurledMaxAbsBinMeanM
              << " uncurled_sinusoid_amplitude_m=" << diagnostic.uncurledSinusoidAmplitudeM
              << " pass=" << (diagnostic.pass ? "true" : "false") << '\n';

    if (!diagnostic.pass)
    {
        for (const std::string &failure : diagnostic.failures)
        {
            std::cerr << "gate: " << failure << '\n';
        }
        return kExitGateFailure;
    }
    return 0;
}
} // namespace

int RunUncurlCheck(std::span<const std::string_view> args)
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
    catch (const std::exception &error)
    {
        std::cerr << "error: " << error.what() << '\n';
        return kExitInvalidInput;
    }
}

RAYLIDAR_REGISTER_COMMAND("uncurl-check",
                          "Masked raw-vs-curled signed range error versus azimuth "
                          "(run mask-build first)",
                          RunUncurlCheck);
