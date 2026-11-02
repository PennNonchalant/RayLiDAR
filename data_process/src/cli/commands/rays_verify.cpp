#include <raylidar/calib/kitti360.h>
#include <raylidar/cli/command.h>
#include <raylidar/cli/commands_phase2.h>
#include <raylidar/common/artifacts.h>
#include <raylidar/common/cli_options.h>
#include <raylidar/common/io.h>
#include <raylidar/gt/cache.h>
#include <raylidar/rays/freeze.h>
#include <raylidar/rays/rays.h>
#include <raylidar/rays/waterfall.h>
#include <raylidar/version.h>

#include <stb_image.h>

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <map>
#include <optional>
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
using raylidar::ParseArgs;
using raylidar::ParsedArgs;
using raylidar::PhaseArtifactsDir;
using raylidar::UsageError;
using raylidar::calib::ComposeTCkRectV;
using raylidar::calib::Kitti360Calib;
using raylidar::calib::LoadKitti360Calib;
using raylidar::calib::LoadPoses;
using raylidar::gt::GtPoint;
using raylidar::gt::ReadGtFrame;
using raylidar::rays::AggregateWaterfalls;
using raylidar::rays::BuildFrameWaterfall;
using raylidar::rays::CameraDropClass;
using raylidar::rays::ClassifyCamera255Point;
using raylidar::rays::FrameWaterfall;
using raylidar::rays::FreezeRaySet;
using raylidar::rays::kRayImageMarginPx;
using raylidar::rays::kRayMaxRangeM;
using raylidar::rays::kRayMinRangeM;
using raylidar::rays::RaySetFreeze;
using raylidar::rays::ReadRayFrame;
using raylidar::rays::ValidateWaterfall;
using raylidar::rays::VerifyRaySet;

constexpr int kExitInvalidInput = 4;
constexpr int kExitGateFailed = 6;

void PrintHelp(std::ostream &out)
{
    out << "Usage: raylidar_cli rays-verify <waterfall|freeze|verify> [options]\n"
        << "\n"
        << "Prove the GT waterfall closes exactly and freeze/verify the ray set.\n"
        << "\n"
        << "Subcommands:\n"
        << "  waterfall  Recompute the exclusive waterfall and write rays/waterfalls.json\n"
        << "  freeze     Validate the waterfall, then write rays/RAY_SET_FREEZE.json\n"
        << "  verify     Recompute every hash and count and compare with the freeze\n"
        << "\n"
        << "Options:\n"
        << "  --config <path>          Phase-2 YAML (default: configs/phase2.yaml)\n"
        << "  --rays-manifest <path>   Manifest inside a rays/ directory\n"
        << "                           (default: <phase_dir>/rays/rays_manifest.json)\n"
        << "  --mask-manifest <path>   Mask manifest\n"
        << "                           (default: <phase_dir>/masks/mask_manifest.json)\n"
        << "  --gt-manifest <path>     GT manifest\n"
        << "                           (default: <phase_dir>/gt/gt_manifest.json)\n"
        << "  --freeze <path>          Freeze file\n"
        << "                           (default: <phase_dir>/rays/RAY_SET_FREEZE.json)\n"
        << "  --help                   Show this help\n"
        << "\n"
        << "Exit codes: 0 success, 2 usage, 4 invalid/missing input,\n"
        << "            6 waterfall/hash/count failure.\n";
}

void Require(const bool condition, const std::string &message, const int exitCode)
{
    if (!condition)
    {
        throw IoError(message, exitCode);
    }
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

double NodeDouble(const YAML::Node &node, const char *key, const double fallback)
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

int NodeInt(const YAML::Node &node, const char *key, const int fallback)
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

std::string FrameName(const int frame)
{
    char buffer[16] = {};
    std::snprintf(buffer, sizeof(buffer), "%010d", frame);
    return std::string(buffer);
}

std::vector<int> FramesFromManifest(const nlohmann::json &manifest, const std::string &context)
{
    Require(manifest.is_object() && manifest.contains("window") && manifest.at("window").is_array() &&
                !manifest.at("window").empty(),
            context + ": missing non-empty window", kExitInvalidInput);
    std::vector<int> frames;
    frames.reserve(manifest.at("window").size());
    for (const nlohmann::json &value : manifest.at("window"))
    {
        Require(value.is_number_integer(), context + ": window entries must be integers", kExitInvalidInput);
        frames.push_back(value.get<int>());
    }
    return frames;
}

std::map<int, const nlohmann::json *> FrameEntries(const nlohmann::json &manifest, const std::string &context)
{
    Require(manifest.is_object() && manifest.contains("frames") && manifest.at("frames").is_array(),
            context + ": missing frames array", kExitInvalidInput);
    std::map<int, const nlohmann::json *> entries;
    for (const nlohmann::json &entry : manifest.at("frames"))
    {
        Require(entry.is_object() && entry.contains("frame") && entry.at("frame").is_number_integer(),
                context + ": every frame entry needs an integer frame", kExitInvalidInput);
        entries.emplace(entry.at("frame").get<int>(), &entry);
    }
    return entries;
}

std::pair<int, int> SemanticsSize(const fs::path &datasetRoot, const std::string &sequence, const std::string &camera,
                                  const int frame)
{
    const fs::path path =
        datasetRoot / "data_2d_semantics" / "train" / sequence / camera / "semantic" / (FrameName(frame) + ".png");
    int width = 0;
    int height = 0;
    int channels = 0;
    if (stbi_info(path.string().c_str(), &width, &height, &channels) == 0)
    {
        throw IoError("cannot read semantic image " + path.string(), kExitInvalidInput);
    }
    Require(width > 0 && height > 0, "semantic image has invalid dimensions: " + path.string(), kExitInvalidInput);
    return {width, height};
}

void WriteJsonAtomic(const fs::path &path, const nlohmann::json &value)
{
    fs::path temporary = path;
    temporary += ".tmp";
    raylidar::WriteJson(temporary, value);
    std::error_code error;
    fs::rename(temporary, path, error);
    if (error)
    {
        std::error_code ignored;
        fs::remove(temporary, ignored);
        throw IoError("cannot replace JSON artifact " + path.string() + ": " + error.message(), kExitInvalidInput);
    }
}

struct Context
{
    fs::path phaseDir;
    fs::path raysManifest;
    fs::path maskManifest;
    fs::path gtManifest;
    fs::path freezePath;
    fs::path configPath;
    std::optional<YAML::Node> config;
};

Context ResolveContext(const ParsedArgs &parsed)
{
    Context context;
    context.phaseDir = PhaseArtifactsDir(2);
    context.configPath = parsed.Get("config", "configs/phase2.yaml");

    std::error_code error;
    const bool configPresent = fs::exists(context.configPath, error);
    if (parsed.Has("config") || configPresent)
    {
        context.config = raylidar::ReadYaml(context.configPath);
        if (context.config.has_value() && context.config->IsMap())
        {
            const YAML::Node artifacts = (*context.config)["artifacts"];
            if (artifacts && artifacts.IsMap())
            {
                const std::string phaseDir = NodeString(artifacts, "phase_dir", "");
                if (!phaseDir.empty())
                {
                    context.phaseDir = phaseDir;
                }
            }
        }
    }

    context.raysManifest = parsed.Get("rays-manifest", "");
    if (context.raysManifest.empty())
    {
        context.raysManifest = context.phaseDir / "rays" / "rays_manifest.json";
    }
    context.maskManifest = parsed.Get("mask-manifest", "");
    if (context.maskManifest.empty())
    {
        context.maskManifest = context.phaseDir / "masks" / "mask_manifest.json";
    }
    context.gtManifest = parsed.Get("gt-manifest", "");
    if (context.gtManifest.empty())
    {
        context.gtManifest = context.phaseDir / "gt" / "gt_manifest.json";
    }
    context.freezePath = parsed.Get("freeze", "");
    if (context.freezePath.empty())
    {
        context.freezePath = context.phaseDir / "rays" / "RAY_SET_FREEZE.json";
    }
    return context;
}

const YAML::Node &RequireConfig(const Context &context)
{
    Require(context.config.has_value() && context.config->IsMap(),
            context.configPath.string() + ": a phase-2 YAML mapping is required", kExitInvalidInput);
    return *context.config;
}

nlohmann::json ComputeWaterfalls(const Context &context)
{
    const YAML::Node &config = RequireConfig(context);
    const std::string datasetRoot = NodeString(config, "dataset_root", "");
    const std::string sequence = NodeString(config, "sequence", "");
    Require(!datasetRoot.empty(), context.configPath.string() + ": dataset_root is not configured", kExitInvalidInput);
    Require(!sequence.empty(), context.configPath.string() + ": sequence is not configured", kExitInvalidInput);

    const YAML::Node raysConfig = config["rays"];
    Require(raysConfig && raysConfig.IsMap(), context.configPath.string() + ": missing rays section",
            kExitInvalidInput);
    Require(std::abs(NodeDouble(raysConfig, "r_min_m", -1.0) - kRayMinRangeM) < 1.0e-12,
            context.configPath.string() + ": rays.r_min_m must match the frozen value", kExitInvalidInput);
    Require(std::abs(NodeDouble(raysConfig, "r_max_m", -1.0) - kRayMaxRangeM) < 1.0e-12,
            context.configPath.string() + ": rays.r_max_m must match the frozen value", kExitInvalidInput);
    Require(NodeInt(raysConfig, "image_margin_px", -1) == kRayImageMarginPx,
            context.configPath.string() + ": rays.image_margin_px must match the frozen value", kExitInvalidInput);

    const nlohmann::json rays = raylidar::ReadJson(context.raysManifest);
    const nlohmann::json masks = raylidar::ReadJson(context.maskManifest);
    const nlohmann::json gt = raylidar::ReadJson(context.gtManifest);

    Require(rays.contains("sequence") && rays.at("sequence").is_string() &&
                rays.at("sequence").get<std::string>() == sequence,
            context.raysManifest.string() + ": sequence disagrees with the config", kExitInvalidInput);
    Require(masks.contains("sequence") && masks.at("sequence").is_string() &&
                masks.at("sequence").get<std::string>() == sequence,
            context.maskManifest.string() + ": sequence disagrees with the config", kExitInvalidInput);

    const std::vector<int> frames = FramesFromManifest(rays, context.raysManifest.string());
    const auto rayEntries = FrameEntries(rays, context.raysManifest.string());
    const auto maskEntries = FrameEntries(masks, context.maskManifest.string());
    const auto gtEntries = FrameEntries(gt, context.gtManifest.string());

    const Kitti360Calib calib = LoadKitti360Calib(fs::path(datasetRoot) / "calibration");
    const std::map<int, Eigen::Matrix4d> poses =
        LoadPoses(fs::path(datasetRoot) / "data_poses" / sequence / "poses.txt");
    const std::vector<std::string> cameras = [&config, &context]() {
        std::vector<std::string> values;
        const YAML::Node node = config["cameras"];
        if (node && node.IsSequence())
        {
            for (const YAML::Node &value : node)
            {
                if (value.IsScalar())
                {
                    values.push_back(value.as<std::string>());
                }
            }
        }
        Require(values.size() >= 2U && values[0] == "image_00" && values[1] == "image_01",
                context.configPath.string() + ": cameras must start with image_00 and image_01", kExitInvalidInput);
        return values;
    }();

    std::vector<FrameWaterfall> waterfalls;
    waterfalls.reserve(frames.size());
    bool allExact = true;

    for (const int frame : frames)
    {
        const auto pose = poses.find(frame);
        Require(pose != poses.end(), "frame " + std::to_string(frame) + " has no exact pose", kExitInvalidInput);
        const auto rayEntry = rayEntries.find(frame);
        Require(rayEntry != rayEntries.end(),
                context.raysManifest.string() + ": missing frame " + std::to_string(frame), kExitInvalidInput);
        const auto maskEntry = maskEntries.find(frame);
        Require(maskEntry != maskEntries.end(),
                context.maskManifest.string() + ": missing frame " + std::to_string(frame), kExitInvalidInput);
        const auto gtEntry = gtEntries.find(frame);
        Require(gtEntry != gtEntries.end(), context.gtManifest.string() + ": missing frame " + std::to_string(frame),
                kExitInvalidInput);

        const nlohmann::json &maskDocument = *maskEntry->second;
        Require(maskDocument.contains("counts") && maskDocument.at("counts").is_object(),
                context.maskManifest.string() + ": frame " + std::to_string(frame) + " has no counts",
                kExitInvalidInput);
        Require(maskDocument.contains("point_count") && maskDocument.at("point_count").is_number_integer(),
                context.maskManifest.string() + ": frame " + std::to_string(frame) + " has no point_count",
                kExitInvalidInput);
        const std::uint64_t raw = maskDocument.at("point_count").get<std::uint64_t>();
        const std::uint64_t keptStatic = maskDocument.at("counts").at("kept_static").get<std::uint64_t>();

        // Recompute the rays actually on disk; never trust the manifest count.
        const nlohmann::json &rayDocument = *rayEntry->second;
        std::string rayFileName;
        if (rayDocument.contains("files") && rayDocument.at("files").is_object() &&
            rayDocument.at("files").contains("ray") && rayDocument.at("files").at("ray").is_string())
        {
            rayFileName = rayDocument.at("files").at("ray").get<std::string>();
        }
        else
        {
            rayFileName = "frame_" + FrameName(frame) + ".bin";
        }
        const fs::path rayPath = context.raysManifest.parent_path() / rayFileName;
        int headerFrame = -1;
        const std::size_t validRays = ReadRayFrame(rayPath, &headerFrame).size();
        Require(headerFrame == frame, rayPath.string() + ": header frame disagrees with the manifest", kExitGateFailed);
        if (rayDocument.contains("n_rays") && rayDocument.at("n_rays").is_number_integer())
        {
            Require(rayDocument.at("n_rays").get<std::uint64_t>() == validRays,
                    rayPath.string() + ": manifest n_rays disagrees with the ray file", kExitGateFailed);
        }

        std::map<std::string, Eigen::Matrix4d> transforms;
        std::map<std::string, Eigen::Matrix3d> intrinsics;
        std::map<std::string, std::pair<int, int>> imageSizes;
        for (const std::string &camera : cameras)
        {
            transforms[camera] = ComposeTCkRectV(calib, pose->second, camera);
            intrinsics[camera] = calib.K_rect.at(camera);
            imageSizes[camera] = SemanticsSize(fs::path(datasetRoot), sequence, camera, frame);
        }

        const fs::path gtPath = context.gtManifest.parent_path() / ("frame_" + FrameName(frame) + ".bin");
        const std::vector<GtPoint> points = ReadGtFrame(gtPath);

        std::uint64_t camera255 = 0;
        std::uint64_t noCameraFov = 0;
        std::uint64_t rangeOutOfDomain = 0;
        for (const GtPoint &point : points)
        {
            if (point.camera != 255U)
            {
                continue;
            }
            ++camera255;
            const Eigen::Vector3d p(point.p_L.x(), point.p_L.y(), point.p_L.z());
            const CameraDropClass drop = ClassifyCamera255Point(p, transforms, intrinsics, imageSizes, kRayMinRangeM,
                                                                kRayMaxRangeM, kRayImageMarginPx);
            if (drop == CameraDropClass::NoCameraFov)
            {
                ++noCameraFov;
            }
            else
            {
                ++rangeOutOfDomain;
            }
        }
        Require(noCameraFov + rangeOutOfDomain == camera255,
                "frame " + std::to_string(frame) + ": camera-255 points do not partition into waterfall reasons",
                kExitGateFailed);

        const FrameWaterfall waterfall = BuildFrameWaterfall(frame, raw, maskDocument.at("counts"), keptStatic,
                                                             validRays, noCameraFov, rangeOutOfDomain);
        std::string reason;
        if (!ValidateWaterfall(waterfall, &reason))
        {
            allExact = false;
            std::cerr << "frame " << frame << ": " << reason << '\n';
        }
        waterfalls.push_back(waterfall);

        std::cout << "frame=" << frame << " raw=" << waterfall.raw << " valid=" << waterfall.valid
                  << " no_camera_fov=" << waterfall.no_camera_fov
                  << " range_out_of_domain=" << waterfall.range_out_of_domain << '\n';
    }

    nlohmann::json document = AggregateWaterfalls(waterfalls);
    document["sequence"] = sequence;
    document["config_sha256"] = raylidar::Sha256File(context.configPath);
    document["generated_at_utc"] = raylidar::UtcNowIso8601();
    document["window"] = frames;
    const fs::path output = context.raysManifest.parent_path() / "waterfalls.json";
    raylidar::EnsureDir(output.parent_path());
    WriteJsonAtomic(output, document);

    if (!allExact || !document["totals"]["exact"].get<bool>())
    {
        throw IoError("waterfall did not close exactly; see " + output.string(), kExitGateFailed);
    }
    return document;
}

int RunWaterfall(const Context &context)
{
    const nlohmann::json document = ComputeWaterfalls(context);
    const fs::path output = context.raysManifest.parent_path() / "waterfalls.json";
    std::cout << "wrote " << output.string() << " raw=" << document["totals"]["raw"].get<std::int64_t>()
              << " valid=" << document["totals"]["valid"].get<std::int64_t>() << '\n';
    return 0;
}

int RunFreeze(const Context &context)
{
    static_cast<void>(ComputeWaterfalls(context));
    const RaySetFreeze freeze =
        FreezeRaySet(context.raysManifest, context.configPath, context.freezePath, raylidar::GetVersion());
    std::cout << "frozen ray_set_sha256=" << freeze.ray_set_sha256 << " manifest_sha256=" << freeze.manifest_sha256
              << '\n';
    return 0;
}

int RunVerify(const Context &context)
{
    const nlohmann::json result = VerifyRaySet(context.raysManifest, context.freezePath);
    if (result["ok"].get<bool>())
    {
        std::cout << "OK ray_set_sha256=" << result["expected"].get<std::string>() << '\n';
        return 0;
    }
    std::cerr << "MISMATCH expected=" << result["expected"].get<std::string>()
              << " actual=" << result["actual"].get<std::string>() << '\n';
    for (const nlohmann::json &frame : result["mismatched_frames"])
    {
        std::cerr << "  mismatched frame " << frame << '\n';
    }
    for (const nlohmann::json &mismatch : result["count_mismatches"])
    {
        std::cerr << "  " << mismatch.get<std::string>() << '\n';
    }
    return kExitGateFailed;
}

int Execute(std::span<const std::string_view> args)
{
    const ParsedArgs parsed =
        ParseArgs(args, {"config", "rays-manifest", "mask-manifest", "gt-manifest", "freeze"}, {"help"});
    if (parsed.Flag("help"))
    {
        PrintHelp(std::cout);
        return 0;
    }

    const std::vector<std::string> positional = parsed.GetAll("");
    if (positional.empty())
    {
        std::cerr << "error: missing rays-verify subcommand"
                     " (expected 'waterfall', 'freeze', or 'verify')\n\n";
        PrintHelp(std::cerr);
        return 2;
    }
    const std::string &subcommand = positional.front();
    const Context context = ResolveContext(parsed);

    if (subcommand == "waterfall")
    {
        return RunWaterfall(context);
    }
    if (subcommand == "freeze")
    {
        return RunFreeze(context);
    }
    if (subcommand == "verify")
    {
        return RunVerify(context);
    }

    std::cerr << "error: unknown rays-verify subcommand '" << subcommand << "'\n\n";
    PrintHelp(std::cerr);
    return 2;
}
} // namespace

int RunRaysVerify(std::span<const std::string_view> args)
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

RAYLIDAR_REGISTER_COMMAND("rays-verify", "Verify the GT waterfall and freeze/verify the ray set", RunRaysVerify);
