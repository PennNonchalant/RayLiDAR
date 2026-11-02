#include <raylidar/calib/kitti360.h>
#include <raylidar/cli/command.h>
#include <raylidar/cli/commands_phase2.h>
#include <raylidar/common/artifacts.h>
#include <raylidar/common/cli_options.h>
#include <raylidar/common/io.h>
#include <raylidar/gt/cache.h>
#include <raylidar/rays/build.h>
#include <raylidar/rays/occlusion.h>
#include <raylidar/rays/rays.h>

#include <stb_image.h>

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
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
using raylidar::rays::Bucket;
using raylidar::rays::BucketName;
using raylidar::rays::BucketOf;
using raylidar::rays::BuildRays;
using raylidar::rays::ComputeOcclusionFlags;
using raylidar::rays::kRayImageMarginPx;
using raylidar::rays::kRayMaxRangeM;
using raylidar::rays::kRayMinRangeM;
using raylidar::rays::kRayRecordSize;
using raylidar::rays::kRayTMinM;
using raylidar::rays::OcclusionOptions;
using raylidar::rays::Ray;
using raylidar::rays::WriteMaskFile;
using raylidar::rays::WriteRayFrame;

constexpr int kExitInvalidInput = 4;

void PrintHelp(std::ostream &out)
{
    out << "Usage: raylidar_cli rays-build [options]\n"
        << "\n"
        << "Build the frozen GT-defined ray set from the per-frame GT cache.\n"
        << "One camera-origin ray is emitted per valid GT point; primary and\n"
        << "secondary masks plus a GT-only occlusion flag are written per frame.\n"
        << "\n"
        << "Options:\n"
        << "  --config <path>          Phase-2 YAML (default: configs/phase2.yaml)\n"
        << "  --gt-manifest <path>     GT manifest inside a gt/ directory\n"
        << "                           (default: <phase_dir>/gt/gt_manifest.json)\n"
        << "  --rays-manifest <path>   Manifest inside a rays/ directory\n"
        << "                           (default: <phase_dir>/rays/rays_manifest.json)\n"
        << "  --out-dir <path>         Ray directory (default: <phase_dir>/rays)\n"
        << "  --frames <list>          Comma-separated smoke frame ids\n"
        << "  --help                   Show this help\n"
        << "\n"
        << "Exit codes: 0 success, 2 usage, 4 invalid input, 6 gate failed.\n";
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

std::vector<std::string> NodeStrings(const YAML::Node &node, const std::string &source)
{
    Require(node && node.IsSequence() && !node.IsNull() && node.size() > 0U, source + " must be a non-empty sequence",
            kExitInvalidInput);
    std::vector<std::string> values;
    values.reserve(node.size());
    for (const YAML::Node &value : node)
    {
        Require(value.IsScalar(), source + " entries must be scalars", kExitInvalidInput);
        values.push_back(value.as<std::string>());
    }
    return values;
}

std::string FrameName(const int frame)
{
    char buffer[16] = {};
    std::snprintf(buffer, sizeof(buffer), "%010d", frame);
    return std::string(buffer);
}

std::vector<int> ParseFrameList(const std::string &text, const std::string &source)
{
    std::vector<int> frames;
    std::size_t start = 0;
    while (start <= text.size())
    {
        const std::size_t comma = text.find(',', start);
        const std::string token = text.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        if (!token.empty())
        {
            try
            {
                std::size_t consumed = 0;
                const int frame = std::stoi(token, &consumed);
                Require(consumed == token.size(), source + ": invalid frame id '" + token + "'", kExitInvalidInput);
                frames.push_back(frame);
            }
            catch (const std::invalid_argument &)
            {
                throw IoError(source + ": invalid frame id '" + token + "'", kExitInvalidInput);
            }
            catch (const std::out_of_range &)
            {
                throw IoError(source + ": frame id out of range '" + token + "'", kExitInvalidInput);
            }
        }
        if (comma == std::string::npos)
        {
            break;
        }
        start = comma + 1U;
    }
    Require(!frames.empty(), source + ": no frame ids supplied", kExitInvalidInput);
    return frames;
}

/// Read the ascending frame window from a GT manifest.
std::vector<int> FramesFromGtManifest(const fs::path &path, std::string *sequence)
{
    const nlohmann::json document = raylidar::ReadJson(path);
    if (sequence != nullptr && document.contains("sequence") && document["sequence"].is_string())
    {
        *sequence = document["sequence"].get<std::string>();
    }
    Require(document.contains("frames") && document["frames"].is_array(), path.string() + ": missing frames array",
            kExitInvalidInput);
    const nlohmann::json &array = document["frames"];
    Require(!array.empty(), path.string() + ": frames array is empty", kExitInvalidInput);
    if (document.contains("totals") && document["totals"].is_object() && document["totals"].contains("frames"))
    {
        Require(document["totals"]["frames"].get<std::size_t>() == array.size(),
                path.string() + ": totals.frames disagrees with the frames array", kExitInvalidInput);
    }

    std::vector<int> frames;
    frames.reserve(array.size());
    for (const nlohmann::json &entry : array)
    {
        Require(entry.is_object() && entry.contains("frame") && entry["frame"].is_number_integer(),
                path.string() + ": every frame entry needs an integer frame", kExitInvalidInput);
        frames.push_back(entry["frame"].get<int>());
    }
    for (std::size_t index = 1; index < frames.size(); ++index)
    {
        Require(frames[index] > frames[index - 1U], path.string() + ": frames must be strictly ascending",
                kExitInvalidInput);
    }
    return frames;
}

void RequireRange(const YAML::Node &node, const std::string &source, const double lower, const double upper)
{
    Require(node && node.IsSequence() && node.size() == 2U, source + " must be a two-element range", kExitInvalidInput);
    Require(std::abs(node[0].as<double>() - lower) < 1.0e-12 && std::abs(node[1].as<double>() - upper) < 1.0e-12,
            source + " does not match the frozen bucket edges", kExitInvalidInput);
}

void RequireBuckets(const YAML::Node &node, const std::string &source,
                    const std::vector<std::pair<double, double>> &expected)
{
    Require(node && node.IsSequence() && node.size() == expected.size(),
            source + " does not match the frozen bucket layout", kExitInvalidInput);
    for (std::size_t index = 0; index < expected.size(); ++index)
    {
        RequireRange(node[index], source + "[" + std::to_string(index) + "]", expected[index].first,
                     expected[index].second);
    }
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

nlohmann::json BucketCounts(const std::array<std::uint64_t, 4> &counts)
{
    nlohmann::json document = nlohmann::json::object();
    document["0_20"] = counts[0];
    document["20_40"] = counts[1];
    document["40_50"] = counts[2];
    document["50_60"] = counts[3];
    return document;
}

int Execute(std::span<const std::string_view> args)
{
    const ParsedArgs parsed =
        ParseArgs(args, {"config", "gt-manifest", "rays-manifest", "out-dir", "frames"}, {"help"});
    if (parsed.Flag("help"))
    {
        PrintHelp(std::cout);
        return 0;
    }

    const std::string configPath = parsed.Get("config", "configs/phase2.yaml");
    const YAML::Node config = raylidar::ReadYaml(configPath);
    Require(config && config.IsMap(), configPath + ": expected a YAML mapping", kExitInvalidInput);

    const std::string datasetRoot = NodeString(config, "dataset_root", "");
    const std::string sequence = NodeString(config, "sequence", "");
    Require(!datasetRoot.empty(), configPath + ": dataset_root is not configured", kExitInvalidInput);
    Require(!sequence.empty(), configPath + ": sequence is not configured", kExitInvalidInput);

    const std::vector<std::string> cameras = NodeStrings(config["cameras"], configPath + ": cameras");
    Require(cameras.size() >= 2U && cameras[0] == "image_00" && cameras[1] == "image_01",
            configPath + ": cameras must start with image_00 and image_01", kExitInvalidInput);

    fs::path phaseDir = NodeString(config["artifacts"], "phase_dir", "");
    if (phaseDir.empty())
    {
        phaseDir = PhaseArtifactsDir(2);
    }

    const YAML::Node raysConfig = config["rays"];
    Require(raysConfig && raysConfig.IsMap(), configPath + ": missing rays section", kExitInvalidInput);
    Require(std::abs(NodeDouble(raysConfig, "t_min_m", -1.0) - kRayTMinM) < 1.0e-12,
            configPath + ": rays.t_min_m must match the frozen value", kExitInvalidInput);
    Require(std::abs(NodeDouble(raysConfig, "r_min_m", -1.0) - kRayMinRangeM) < 1.0e-12,
            configPath + ": rays.r_min_m must match the frozen value", kExitInvalidInput);
    Require(std::abs(NodeDouble(raysConfig, "r_max_m", -1.0) - kRayMaxRangeM) < 1.0e-12,
            configPath + ": rays.r_max_m must match the frozen value", kExitInvalidInput);
    Require(NodeInt(raysConfig, "image_margin_px", -1) == kRayImageMarginPx,
            configPath + ": rays.image_margin_px must match the frozen value", kExitInvalidInput);
    Require(NodeInt(raysConfig, "record_size", -1) == static_cast<int>(kRayRecordSize),
            configPath + ": rays.record_size must be 64", kExitInvalidInput);
    RequireBuckets(raysConfig["buckets_primary"], configPath + ": rays.buckets_primary",
                   {{0.0, 20.0}, {20.0, 40.0}, {40.0, 50.0}});
    RequireBuckets(raysConfig["buckets_secondary"], configPath + ": rays.buckets_secondary", {{50.0, 60.0}});

    const YAML::Node occlusionConfig = raysConfig["occlusion"];
    Require(occlusionConfig && occlusionConfig.IsMap(), configPath + ": missing rays.occlusion section",
            kExitInvalidInput);
    OcclusionOptions occlusion;
    occlusion.neighborPx = NodeInt(occlusionConfig, "neighbor_px", -1);
    occlusion.sigmaLidarM = NodeDouble(occlusionConfig, "sigma_lidar_m", -1.0);
    occlusion.sigmaMultiple = NodeDouble(occlusionConfig, "sigma_multiple", -1.0);
    occlusion.depthThresholdM = NodeDouble(occlusionConfig, "depth_threshold_m", -1.0);
    Require(occlusion.neighborPx >= 0, configPath + ": rays.occlusion.neighbor_px is invalid", kExitInvalidInput);
    Require(occlusion.sigmaLidarM >= 0.0, configPath + ": rays.occlusion.sigma_lidar_m is invalid", kExitInvalidInput);
    Require(occlusion.sigmaMultiple >= 0.0, configPath + ": rays.occlusion.sigma_multiple is invalid",
            kExitInvalidInput);
    Require(occlusion.depthThresholdM > 0.0, configPath + ": rays.occlusion.depth_threshold_m must be positive",
            kExitInvalidInput);
    const std::vector<std::string> variants =
        NodeStrings(occlusionConfig["variants"], configPath + ": rays.occlusion.variants");
    Require(std::find(variants.begin(), variants.end(), "occluded_excluded") != variants.end() &&
                std::find(variants.begin(), variants.end(), "full") != variants.end(),
            configPath + ": rays.occlusion.variants must include occluded_excluded and full", kExitInvalidInput);

    fs::path outDir = parsed.Get("out-dir", "");
    if (outDir.empty())
    {
        outDir = phaseDir / "rays";
    }

    fs::path gtManifest = parsed.Get("gt-manifest", "");
    if (gtManifest.empty())
    {
        gtManifest = phaseDir / "gt" / "gt_manifest.json";
    }
    Require(gtManifest.parent_path().filename() == "gt", "--gt-manifest must live inside a 'gt' directory",
            kExitInvalidInput);

    fs::path raysManifest = parsed.Get("rays-manifest", "");
    if (raysManifest.empty())
    {
        raysManifest = phaseDir / "rays" / "rays_manifest.json";
    }
    Require(raysManifest.parent_path().filename() == "rays", "--rays-manifest must live inside a 'rays' directory",
            kExitInvalidInput);

    std::string gtSequence;
    std::vector<int> frames;
    const std::string framesText = parsed.Get("frames", "");
    const bool smoke = !framesText.empty();
    if (smoke)
    {
        frames = ParseFrameList(framesText, "--frames");
        raysManifest = raysManifest.parent_path() / "rays_manifest.smoke.json";
    }
    else
    {
        frames = FramesFromGtManifest(gtManifest, &gtSequence);
        Require(gtSequence.empty() || gtSequence == sequence,
                configPath + ": GT manifest sequence disagrees with the config", kExitInvalidInput);
    }
    std::sort(frames.begin(), frames.end());
    frames.erase(std::unique(frames.begin(), frames.end()), frames.end());
    Require(!frames.empty(), "rays-build needs at least one frame", kExitInvalidInput);

    const std::string configSha = raylidar::Sha256File(configPath);
    if (!smoke)
    {
        const nlohmann::json document = raylidar::ReadJson(gtManifest);
        if (document.contains("config_sha256") && document["config_sha256"].is_string())
        {
            Require(document["config_sha256"].get<std::string>() == configSha,
                    gtManifest.string() + ": GT cache was built with a different config (stale transforms)",
                    kExitInvalidInput);
        }
    }

    const Kitti360Calib calib = LoadKitti360Calib(fs::path(datasetRoot) / "calibration");
    for (const std::string &camera : {cameras[0], cameras[1]})
    {
        Require(calib.K_rect.find(camera) != calib.K_rect.end(), "calibration has no rectified K for " + camera,
                kExitInvalidInput);
    }
    const std::map<int, Eigen::Matrix4d> poses =
        LoadPoses(fs::path(datasetRoot) / "data_poses" / sequence / "poses.txt");

    raylidar::EnsureDir(outDir);
    raylidar::EnsureDir(raysManifest.parent_path());

    std::vector<nlohmann::json> frameDocuments;
    frameDocuments.reserve(frames.size());
    std::array<std::uint64_t, 4> totalBuckets{0U, 0U, 0U, 0U};
    std::array<std::uint64_t, 2> totalCameras{0U, 0U};
    std::uint64_t totalRays = 0;
    std::uint64_t totalOccluded = 0;
    std::uint64_t totalOccludedExcluded = 0;

    for (const int frame : frames)
    {
        const auto pose = poses.find(frame);
        Require(pose != poses.end(), "frame " + std::to_string(frame) + " has no exact pose", kExitInvalidInput);

        const fs::path gtPath = gtManifest.parent_path() / ("frame_" + FrameName(frame) + ".bin");
        const std::vector<GtPoint> gtPoints = ReadGtFrame(gtPath);

        // W11: compose each camera transform once per frame and pass the map.
        std::map<std::string, Eigen::Matrix4d> transforms;
        transforms[cameras[0]] = ComposeTCkRectV(calib, pose->second, cameras[0]);
        transforms[cameras[1]] = ComposeTCkRectV(calib, pose->second, cameras[1]);

        std::vector<Ray> rays = BuildRays(gtPoints, frame, transforms);

        const std::pair<int, int> size0 = SemanticsSize(fs::path(datasetRoot), sequence, cameras[0], frame);
        const std::pair<int, int> size1 = SemanticsSize(fs::path(datasetRoot), sequence, cameras[1], frame);

        const std::vector<std::uint8_t> flags0 =
            ComputeOcclusionFlags(gtPoints, rays, 0, transforms[cameras[0]], calib.K_rect.at(cameras[0]), size0.first,
                                  size0.second, occlusion);
        const std::vector<std::uint8_t> flags1 =
            ComputeOcclusionFlags(gtPoints, rays, 1, transforms[cameras[1]], calib.K_rect.at(cameras[1]), size1.first,
                                  size1.second, occlusion);
        for (std::size_t index = 0; index < rays.size(); ++index)
        {
            rays[index].occluded = static_cast<std::uint8_t>(flags0[index] | flags1[index]);
        }

        std::array<std::uint64_t, 4> bucketCounts{0U, 0U, 0U, 0U};
        std::array<std::uint64_t, 2> cameraCounts{0U, 0U};
        std::uint64_t occludedCount = 0;
        std::uint64_t occludedExcluded = 0;
        std::vector<std::uint8_t> primaryMask;
        std::vector<std::uint8_t> secondaryMask;
        std::vector<std::uint8_t> primaryUnoccludedMask;
        primaryMask.reserve(rays.size());
        secondaryMask.reserve(rays.size());
        primaryUnoccludedMask.reserve(rays.size());
        for (const Ray &ray : rays)
        {
            const std::size_t bucket = static_cast<std::size_t>(ray.bucket);
            Require(bucket < bucketCounts.size(), "ray bucket is out of range", kExitInvalidInput);
            const bool primary = ray.bucket != static_cast<std::uint8_t>(Bucket::B50_60);
            ++bucketCounts[bucket];
            if (ray.camera < cameraCounts.size())
            {
                ++cameraCounts[ray.camera];
            }
            if (ray.occluded != 0U)
            {
                ++occludedCount;
            }
            const bool unoccludedPrimary = primary && ray.occluded == 0U;
            if (unoccludedPrimary)
            {
                ++occludedExcluded;
            }
            primaryMask.push_back(primary ? 1U : 0U);
            secondaryMask.push_back(primary ? 0U : 1U);
            primaryUnoccludedMask.push_back(unoccludedPrimary ? 1U : 0U);
        }

        const std::string name = FrameName(frame);
        const fs::path rayPath = outDir / ("frame_" + name + ".bin");
        const fs::path primaryPath = outDir / ("primary_mask_" + name + ".bin");
        const fs::path secondaryPath = outDir / ("secondary_mask_" + name + ".bin");
        const fs::path unoccludedPath = outDir / ("primary_unoccluded_mask_" + name + ".bin");
        WriteRayFrame(rayPath, frame, rays);
        WriteMaskFile(primaryPath, primaryMask);
        WriteMaskFile(secondaryPath, secondaryMask);
        WriteMaskFile(unoccludedPath, primaryUnoccludedMask);

        nlohmann::json entry;
        entry["frame"] = frame;
        entry["n_rays"] = rays.size();
        entry["bucket_counts"] = BucketCounts(bucketCounts);
        entry["camera_counts"] = {{"image_00", cameraCounts[0]}, {"image_01", cameraCounts[1]}};
        entry["occluded_count"] = occludedCount;
        entry["occluded_excluded_count"] = occludedExcluded;
        entry["full_count"] = rays.size();
        entry["sha256"] = {
            {"ray", raylidar::Sha256File(rayPath)},
            {"primary_mask", raylidar::Sha256File(primaryPath)},
            {"secondary_mask", raylidar::Sha256File(secondaryPath)},
            {"primary_unoccluded_mask", raylidar::Sha256File(unoccludedPath)},
        };
        entry["files"] = {
            {"ray", rayPath.filename().string()},
            {"primary_mask", primaryPath.filename().string()},
            {"secondary_mask", secondaryPath.filename().string()},
            {"primary_unoccluded_mask", unoccludedPath.filename().string()},
        };

        const fs::path metaPath = outDir / ("frame_" + name + "_meta.json");
        WriteJsonAtomic(metaPath, entry);
        frameDocuments.push_back(std::move(entry));

        for (std::size_t index = 0; index < bucketCounts.size(); ++index)
        {
            totalBuckets[index] += bucketCounts[index];
        }
        totalCameras[0] += cameraCounts[0];
        totalCameras[1] += cameraCounts[1];
        totalRays += rays.size();
        totalOccluded += occludedCount;
        totalOccludedExcluded += occludedExcluded;

        std::cout << "frame=" << frame << " rays=" << rays.size() << " occluded=" << occludedCount
                  << " primary_unoccluded=" << occludedExcluded << '\n';
    }

    nlohmann::json totals;
    totals["frames"] = frameDocuments.size();
    totals["n_rays"] = totalRays;
    totals["bucket_counts"] = BucketCounts(totalBuckets);
    totals["camera_counts"] = {{"image_00", totalCameras[0]}, {"image_01", totalCameras[1]}};
    totals["occluded_count"] = totalOccluded;
    totals["occluded_excluded_count"] = totalOccludedExcluded;
    totals["full_count"] = totalRays;

    nlohmann::json manifest;
    manifest["schema_version"] = 1;
    manifest["sequence"] = sequence;
    manifest["window"] = frames;
    manifest["generated_at_utc"] = raylidar::UtcNowIso8601();
    manifest["frames"] = std::move(frameDocuments);
    manifest["totals"] = std::move(totals);
    manifest["config_sha256"] = configSha;
    WriteJsonAtomic(raysManifest, manifest);

    if (smoke)
    {
        std::cerr << "warning: smoke run wrote " << raysManifest.filename().string() << "; this is not final\n";
    }
    return 0;
}
} // namespace

int RunRaysBuild(std::span<const std::string_view> args)
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

RAYLIDAR_REGISTER_COMMAND("rays-build", "Build the frozen GT-defined ray set and its masks", RunRaysBuild);
