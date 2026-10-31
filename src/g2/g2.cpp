#include <raylidar/g2/g2.h>

#include <raylidar/calib/kitti360.h>
#include <raylidar/common/io.h>
#include <raylidar/prereg/prereg.h>
#include <raylidar/rays/freeze.h>
#include <raylidar/rays/rays.h>
#include <raylidar/rays/waterfall.h>

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
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace raylidar::g2
{
namespace
{
namespace fs = std::filesystem;

constexpr int kWindowFrames = 60;
constexpr std::size_t kDomainSamplePerFrame = 1000;
constexpr double kDirectionNormTolerance = 1e-4;
constexpr double kPoseReprojectionTolerance = 1e-3;
constexpr double kRangeReprojectionTolerance = 1e-3;

const std::array<const char *, 9> kCheckIds = {"prereg", "segment", "uncurl", "masks", "gt",
                                               "rays",    "waterfall", "hash", "domain"};
const std::array<const char *, 4> kBucketKeys = {"0_20", "20_40", "40_50", "50_60"};

[[noreturn]] void Fail(const std::string &message)
{
    throw std::runtime_error(message);
}

void Require(const bool condition, const std::string &message)
{
    if (!condition)
    {
        Fail(message);
    }
}

std::string FrameName(const int frame)
{
    char buffer[16] = {};
    std::snprintf(buffer, sizeof(buffer), "%010d", frame);
    return std::string(buffer);
}

std::string NodeString(const YAML::Node &node, const char *key, const std::string &fallback = {})
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
        return value.as<std::string>();
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

std::vector<std::string> NodeStrings(const YAML::Node &node)
{
    std::vector<std::string> values;
    if (!node || !node.IsSequence())
    {
        return values;
    }
    for (const YAML::Node &value : node)
    {
        if (value.IsScalar())
        {
            values.push_back(value.as<std::string>());
        }
    }
    return values;
}

const nlohmann::json &RequireObject(const nlohmann::json &document, const char *key,
                                    const std::string &context)
{
    if (!document.is_object() || !document.contains(key) || !document.at(key).is_object())
    {
        Fail(context + ": missing object '" + key + "'");
    }
    return document.at(key);
}

const nlohmann::json &RequireArray(const nlohmann::json &document, const char *key,
                                   const std::string &context)
{
    if (!document.is_object() || !document.contains(key) || !document.at(key).is_array())
    {
        Fail(context + ": missing array '" + key + "'");
    }
    return document.at(key);
}

std::string RequireString(const nlohmann::json &document, const char *key,
                          const std::string &context)
{
    if (!document.is_object() || !document.contains(key) || !document.at(key).is_string())
    {
        Fail(context + ": missing string '" + key + "'");
    }
    return document.at(key).get<std::string>();
}

std::int64_t RequireInt(const nlohmann::json &document, const char *key,
                        const std::string &context)
{
    if (!document.is_object() || !document.contains(key) || !document.at(key).is_number_integer())
    {
        Fail(context + ": missing integer '" + key + "'");
    }
    return document.at(key).get<std::int64_t>();
}

std::int64_t RequireCount(const nlohmann::json &document, const char *key,
                          const std::string &context)
{
    const std::int64_t value = RequireInt(document, key, context);
    Require(value >= 0, context + ": negative count '" + key + "'");
    return value;
}

bool GetBool(const nlohmann::json &document, const char *key, const bool fallback)
{
    if (!document.is_object())
    {
        return fallback;
    }
    const auto found = document.find(key);
    if (found == document.end() || !found->is_boolean())
    {
        return fallback;
    }
    return found->get<bool>();
}

std::vector<int> IntArray(const nlohmann::json &array, const std::string &context)
{
    Require(array.is_array(), context + " must be an array");
    std::vector<int> values;
    values.reserve(array.size());
    for (const nlohmann::json &value : array)
    {
        Require(value.is_number_integer(), context + " entries must be integers");
        values.push_back(value.get<int>());
    }
    return values;
}

struct Context
{
    fs::path phaseDir;
    fs::path configPath;
    fs::path projectRoot;
    fs::path datasetRoot;
    fs::path decisionsPath;
    fs::path protocolPath;
    fs::path sidecarPath;
    fs::path sidecarJsonPath;
    std::string sequence;
    std::vector<std::string> cameras;
    double rMinM = rays::kRayMinRangeM;
    double rMaxM = rays::kRayMaxRangeM;
    int marginPx = rays::kRayImageMarginPx;
    bool requireBothCameras = true;
    int minValidRays = 20000;
    double gtMinCount = 50000.0;
    double gtMaxCount = 150000.0;
    int gtMinPoints = 20000;
    bool egoBoxConfigured = false;
    std::vector<int> selectedFrames;
};

Context LoadContext(const fs::path &phaseDir, const fs::path &configPath)
{
    Context context;
    context.configPath = configPath;
    context.phaseDir = phaseDir;

    const YAML::Node config = ReadYaml(configPath);
    Require(config && config.IsMap(), configPath.string() + ": expected a YAML mapping");

    context.projectRoot = configPath.parent_path().parent_path();
    if (context.projectRoot.empty())
    {
        context.projectRoot = fs::path(".");
    }
    context.datasetRoot = NodeString(config, "dataset_root");
    context.sequence = NodeString(config, "sequence");
    Require(!context.datasetRoot.empty(), configPath.string() + ": dataset_root is not configured");
    Require(!context.sequence.empty(), configPath.string() + ": sequence is not configured");

    context.cameras = NodeStrings(config["cameras"]);
    Require(context.cameras.size() >= 2U && context.cameras[0] == "image_00" &&
                context.cameras[1] == "image_01",
            configPath.string() + ": cameras must start with image_00 and image_01");

    if (context.phaseDir.empty())
    {
        const std::string configured = NodeString(config["artifacts"], "phase_dir");
        Require(!configured.empty(), configPath.string() + ": artifacts.phase_dir is not configured");
        context.phaseDir = configured;
    }

    const YAML::Node raysConfig = config["rays"];
    context.rMinM = NodeDouble(raysConfig, "r_min_m", rays::kRayMinRangeM);
    context.rMaxM = NodeDouble(raysConfig, "r_max_m", rays::kRayMaxRangeM);
    context.marginPx = NodeInt(raysConfig, "image_margin_px", rays::kRayImageMarginPx);
    Require(std::abs(context.rMinM - rays::kRayMinRangeM) < 1e-12,
            configPath.string() + ": rays.r_min_m must match the frozen value");
    Require(std::abs(context.rMaxM - rays::kRayMaxRangeM) < 1e-12,
            configPath.string() + ": rays.r_max_m must match the frozen value");
    Require(context.marginPx == rays::kRayImageMarginPx,
            configPath.string() + ": rays.image_margin_px must match the frozen value");

    const YAML::Node g2Config = config["g2"];
    context.requireBothCameras = true;
    if (g2Config && g2Config.IsMap())
    {
        const YAML::Node requireBoth = g2Config["require_image_00_and_01"];
        if (requireBoth && requireBoth.IsScalar())
        {
            context.requireBothCameras = requireBoth.as<bool>();
        }
        context.minValidRays = NodeInt(g2Config, "min_valid_rays_per_frame", 20000);
    }
    Require(context.minValidRays > 0,
            configPath.string() + ": g2.min_valid_rays_per_frame must be positive");

    const YAML::Node gtConfig = config["gt_cache"];
    context.gtMinPoints = NodeInt(gtConfig, "min_points_per_frame", 20000);
    const YAML::Node expected = gtConfig ? gtConfig["expected_points_range"] : YAML::Node();
    if (expected && expected.IsSequence() && expected.size() == 2U)
    {
        context.gtMinCount = expected[0].as<double>();
        context.gtMaxCount = expected[1].as<double>();
    }

    const YAML::Node maskConfig = config["mask"];
    const YAML::Node egoRoi = maskConfig ? maskConfig["ego_roi_velodyne_m"] : YAML::Node();
    context.egoBoxConfigured =
        egoRoi && egoRoi.IsMap() && egoRoi["x"] && egoRoi["y"] && egoRoi["z"];

    context.decisionsPath = context.projectRoot / "docs" / "decisions.md";
    context.protocolPath = context.projectRoot / "configs" / "protocol.yaml";
    context.sidecarPath = context.phaseDir.parent_path() / "01" / "decisions.sha256";
    context.sidecarJsonPath = context.phaseDir.parent_path() / "01" / "decisions.sha256.json";
    return context;
}

std::vector<int> RequireSelected(const Context &context)
{
    Require(!context.selectedFrames.empty(), "segment evidence is missing");
    return context.selectedFrames;
}

void RequireAscending(const std::vector<int> &frames, const std::string &context)
{
    for (std::size_t index = 1; index < frames.size(); ++index)
    {
        Require(frames[index] > frames[index - 1], context + ": frames must be strictly ascending");
    }
}

void RequireSameFrames(const std::vector<int> &actual, const std::vector<int> &expected,
                       const std::string &context)
{
    Require(actual == expected, context + ": frame set disagrees with selected.frames");
}

std::map<int, const nlohmann::json *> FrameIndex(const nlohmann::json &manifest,
                                                 const std::string &context)
{
    const nlohmann::json &frames = RequireArray(manifest, "frames", context);
    std::map<int, const nlohmann::json *> entries;
    for (const nlohmann::json &entry : frames)
    {
        const int frame = static_cast<int>(RequireInt(entry, "frame", context + " frame"));
        Require(entries.emplace(frame, &entry).second,
                context + ": duplicate frame " + std::to_string(frame));
    }
    return entries;
}

nlohmann::json CheckPrereg(const Context &context, nlohmann::json &report)
{
    const nlohmann::json verified =
        prereg::VerifyDecisions(context.decisionsPath, context.sidecarPath);
    Require(verified.at("ok").get<bool>(),
            "decisions SHA-256 does not match the frozen sidecar");

    const nlohmann::json companion = ReadJson(context.sidecarJsonPath);
    const std::string protocolSha = Sha256File(context.protocolPath);
    Require(RequireString(companion, "protocol_sha256", "decisions.sha256.json") == protocolSha,
            "protocol_sha256 does not match configs/protocol.yaml");

    const YAML::Node protocol = ReadYaml(context.protocolPath);
    Require(protocol && protocol.IsMap(), "configs/protocol.yaml: expected a YAML mapping");
    const std::string runId = NodeString(protocol, "run_id");
    Require(!runId.empty(), "configs/protocol.yaml: run_id is empty");
    Require(NodeInt(protocol["frame_window"], "window_frames", -1) == kWindowFrames,
            "configs/protocol.yaml: frame_window.window_frames must be 60");
    Require(NodeInt(protocol["frame_window"], "step_frames", -1) == 10,
            "configs/protocol.yaml: frame_window.step_frames must be 10");
    Require(NodeString(protocol["sequence_policy"], "tie_break") == "earliest_window",
            "configs/protocol.yaml: sequence_policy.tie_break must be earliest_window");
    const std::vector<std::string> sources = NodeStrings(protocol["dynamic_scoring"]["sources"]);
    Require(sources == std::vector<std::string>{"3d_bboxes", "2d_semantics"},
            "configs/protocol.yaml: dynamic_scoring.sources must be [3d_bboxes, 2d_semantics]");

    report["run_id"] = runId;
    report["decisions_sha256"] = verified.at("actual").get<std::string>();
    report["protocol_sha256"] = protocolSha;
    report["frozen_at_utc"] = verified.value("frozen_at_utc", std::string{});

    return {{"run_id", runId},
            {"decisions_sha256", report["decisions_sha256"]},
            {"protocol_sha256", protocolSha},
            {"frozen_at_utc", report["frozen_at_utc"]}};
}

nlohmann::json CheckSegment(const Context &context)
{
    const fs::path path = context.phaseDir / "segment" / "segment_manifest.json";
    const nlohmann::json manifest = ReadJson(path);
    const std::string manifestContext = path.string();
    Require(RequireString(manifest, "selection_mode", manifestContext) == "full",
            manifestContext + ": selection_mode must be full");
    const nlohmann::json &selected = RequireObject(manifest, "selected", manifestContext);
    const std::string selectedContext = manifestContext + " selected";
    const std::vector<int> frames =
        IntArray(RequireArray(selected, "frames", selectedContext), "selected.frames");
    const std::int64_t frameCount = RequireInt(selected, "frame_count", manifestContext);
    Require(frames.size() == static_cast<std::size_t>(kWindowFrames),
            path.string() + ": selected.frames must have exactly 60 entries");
    Require(frameCount == static_cast<std::int64_t>(frames.size()),
            path.string() + ": selected.frame_count disagrees with selected.frames");
    RequireAscending(frames, path.string());

    const nlohmann::json &candidates =
        RequireArray(manifest, "candidate_windows", manifestContext);
    Require(!candidates.empty(), manifestContext + ": candidate_windows is empty");
    const std::string rationale = RequireString(selected, "rationale", manifestContext);
    Require(!rationale.empty(), path.string() + ": selected.rationale is empty");

    bool ledger = false;
    for (const char *key : {"failures", "dropped_frames", "dropped"})
    {
        if (manifest.contains(key) && manifest.at(key).is_array())
        {
            ledger = true;
        }
    }
    Require(ledger, path.string() + ": missing dropped-frame ledger");

    return {{"frames", frames},
            {"frame_count", frameCount},
            {"candidate_windows", candidates.size()},
            {"rationale", rationale}};
}

nlohmann::json CheckUncurl(const Context &context)
{
    const std::vector<int> frames = RequireSelected(context);
    const fs::path path = context.phaseDir / "uncurl" / "uncurl_report.json";
    const nlohmann::json report = ReadJson(path);
    Require(GetBool(report, "pass", false), path.string() + ": pass is not true");
    const std::vector<int> reported =
        IntArray(RequireArray(report, "frames", path.string()), "uncurl frames");
    RequireSameFrames(reported, frames, path.string());

    const std::string maskDir = RequireString(report, "mask_dir", path.string());
    Require(!maskDir.empty(), path.string() + ": mask_dir is empty");
    std::error_code error;
    Require(fs::is_directory(maskDir, error), path.string() + ": mask_dir does not exist");

    const std::string reportContext = path.string();
    const auto checkVariant = [&reportContext](const nlohmann::json &document,
                                               const std::string &name) {
        const std::string variantContext = reportContext + " " + name;
        const nlohmann::json &bins = RequireArray(document, "bins", variantContext);
        Require(!bins.empty(), reportContext + ": " + name + " bins are empty");
        for (const nlohmann::json &bin : bins)
        {
            Require(RequireCount(bin, "count", variantContext) >= 0,
                    reportContext + ": negative bin count");
        }
        const auto peak = document.find("peak_to_peak_m");
        Require(peak != document.end() && peak->is_number(),
                reportContext + ": " + name + " peak_to_peak_m is missing");
    };
    checkVariant(RequireObject(report, "raw", reportContext), "raw");
    checkVariant(RequireObject(report, "uncurled", reportContext), "uncurled");

    for (const char *key : {"rawPeakToPeakM", "uncurledPeakToPeakM"})
    {
        const auto found = report.find(key);
        Require(found != report.end() && found->is_number(),
                path.string() + ": missing " + key);
    }

    return {{"frames", reported.size()},
            {"mask_dir", maskDir},
            {"raw_peak_to_peak_m", report.at("rawPeakToPeakM")},
            {"uncurled_peak_to_peak_m", report.at("uncurledPeakToPeakM")}};
}

nlohmann::json CheckMasks(const Context &context)
{
    const std::vector<int> frames = RequireSelected(context);
    const fs::path path = context.phaseDir / "masks" / "mask_manifest.json";
    const nlohmann::json manifest = ReadJson(path);
    const std::string manifestContext = path.string();
    const nlohmann::json &entries = RequireArray(manifest, "frames", manifestContext);
    Require(entries.size() == static_cast<std::size_t>(kWindowFrames),
            manifestContext + ": must contain exactly 60 frames");

    std::error_code error;
    std::vector<int> maskFrames;
    maskFrames.reserve(entries.size());
    for (const nlohmann::json &entry : entries)
    {
        const int frame = static_cast<int>(RequireInt(entry, "frame", path.string()));
        maskFrames.push_back(frame);
        const std::string context = path.string() + " frame " + std::to_string(frame);

        const nlohmann::json &counts = RequireObject(entry, "counts", context);
        Require(counts.contains("ego_roi") && counts.at("ego_roi").is_number_integer(),
                context + ": counts.ego_roi is missing");
        Require(entry.contains("residual_dynamic") && entry.at("residual_dynamic").is_object(),
                context + ": residual_dynamic is missing");

        const nlohmann::json &appliesTo = RequireArray(entry, "applies_to", context);
        std::set<std::string> labels;
        for (const nlohmann::json &value : appliesTo)
        {
            if (value.is_string())
            {
                labels.insert(value.get<std::string>());
            }
        }
        for (const char *label : {"images", "gt", "projections", "evaluation"})
        {
            Require(labels.count(label) == 1U, context + ": applies_to is missing " + label);
        }

        const fs::path frameDir = path.parent_path() / ("frame_" + FrameName(frame));
        for (const char *file : {"points.bin", "image_00_dynamic.png", "image_01_dynamic.png"})
        {
            Require(fs::is_regular_file(frameDir / file, error),
                    context + ": missing " + std::string(file));
        }
    }
    RequireSameFrames(maskFrames, frames, path.string());

    const bool boxMode = manifest.contains("ego_roi_mode") &&
                         manifest.at("ego_roi_mode").is_string() &&
                         manifest.at("ego_roi_mode").get<std::string>() == "box";
    Require(boxMode || context.egoBoxConfigured,
            path.string() + ": ego_roi_mode must be box or mask.ego_roi_velodyne_m configured");

    return {{"frames", entries.size()}, {"ego_roi_configured", context.egoBoxConfigured}};
}

nlohmann::json CheckGt(const Context &context)
{
    const std::vector<int> frames = RequireSelected(context);
    const fs::path path = context.phaseDir / "gt" / "gt_manifest.json";
    const nlohmann::json manifest = ReadJson(path);
    const std::string manifestContext = path.string();
    const nlohmann::json &entries = RequireArray(manifest, "frames", manifestContext);
    Require(entries.size() == static_cast<std::size_t>(kWindowFrames),
            manifestContext + ": must contain exactly 60 frames");

    std::vector<int> gtFrames;
    std::int64_t total = 0;
    for (const nlohmann::json &entry : entries)
    {
        const int frame = static_cast<int>(RequireInt(entry, "frame", path.string()));
        gtFrames.push_back(frame);
        const std::string frameContext = path.string() + " frame " + std::to_string(frame);
        const std::int64_t count = RequireCount(entry, "count", frameContext);
        Require(count >= context.gtMinPoints,
                frameContext + ": count is below min_points_per_frame");
        Require(static_cast<double>(count) >= context.gtMinCount &&
                    static_cast<double>(count) <= context.gtMaxCount,
                frameContext + ": count is outside expected_points_range");

        const std::string stored = RequireString(entry, "sha256", frameContext);
        const fs::path gtPath = path.parent_path() / ("frame_" + FrameName(frame) + ".bin");
        Require(Sha256File(gtPath) == stored, frameContext + ": SHA-256 mismatch");
        total += count;
    }
    RequireSameFrames(gtFrames, frames, path.string());
    return {{"frames", entries.size()}, {"total_points", total}};
}

struct DerivedRayCounts
{
    std::uint64_t nRays = 0;
    std::array<std::uint64_t, 4> buckets{0U, 0U, 0U, 0U};
    std::array<std::uint64_t, 2> cameras{0U, 0U};
    std::uint64_t occluded = 0;
    std::uint64_t unoccludedPrimary = 0;
};

std::vector<std::uint8_t> ReadBytes(const fs::path &path)
{
    std::error_code error;
    Require(fs::is_regular_file(path, error), "missing mask file " + path.string());
    std::ifstream input(path, std::ios::binary);
    Require(input.good(), "cannot read " + path.string());
    return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(input),
                                     std::istreambuf_iterator<char>());
}

std::string FileNameFor(const nlohmann::json &entry, const char *key, const std::string &context)
{
    const nlohmann::json &files = RequireObject(entry, "files", context);
    if (!files.contains(key) || !files.at(key).is_string())
    {
        Fail(context + ": missing files." + std::string(key));
    }
    return files.at(key).get<std::string>();
}

void CompareCount(const std::uint64_t expected, const std::int64_t actual,
                  const std::string &label)
{
    Require(actual >= 0 && static_cast<std::uint64_t>(actual) == expected,
            label + ": stored=" + std::to_string(actual) + " derived=" +
                std::to_string(expected));
}

nlohmann::json CheckRays(const Context &context)
{
    const std::vector<int> frames = RequireSelected(context);
    const fs::path manifestPath = context.phaseDir / "rays" / "rays_manifest.json";
    const nlohmann::json manifest = ReadJson(manifestPath);
    const std::string manifestContext = manifestPath.string();
    const std::vector<int> window =
        IntArray(RequireArray(manifest, "window", manifestContext), "rays window");
    RequireSameFrames(window, frames, manifestContext);
    const nlohmann::json &entries = RequireArray(manifest, "frames", manifestContext);
    Require(entries.size() == static_cast<std::size_t>(kWindowFrames),
            manifestContext + ": must contain exactly 60 frames");

    std::vector<int> rayFrames;
    std::uint64_t totalRays = 0;
    std::uint64_t minRays = std::numeric_limits<std::uint64_t>::max();
    for (const nlohmann::json &entry : entries)
    {
        const int frame = static_cast<int>(RequireInt(entry, "frame", manifestPath.string()));
        rayFrames.push_back(frame);
        const std::string frameContext = "rays frame " + std::to_string(frame);

        const std::string rayName = FileNameFor(entry, "ray", frameContext);
        int headerFrame = -1;
        const std::vector<rays::Ray> rays =
            rays::ReadRayFrame(manifestPath.parent_path() / rayName, &headerFrame);
        Require(headerFrame == frame, frameContext + ": ray header frame mismatch");

        DerivedRayCounts derived;
        derived.nRays = rays.size();
        for (const rays::Ray &ray : rays)
        {
            Require(ray.bucket < derived.buckets.size(), frameContext + ": bucket out of range");
            ++derived.buckets[static_cast<std::size_t>(ray.bucket)];
            Require(ray.camera < derived.cameras.size(), frameContext + ": camera out of range");
            ++derived.cameras[static_cast<std::size_t>(ray.camera)];
            if (ray.occluded != 0U)
            {
                ++derived.occluded;
            }
            if (ray.bucket != static_cast<std::uint8_t>(rays::Bucket::B50_60) &&
                ray.occluded == 0U)
            {
                ++derived.unoccludedPrimary;
            }
        }

        const std::uint64_t nRays =
            static_cast<std::uint64_t>(RequireCount(entry, "n_rays", frameContext));
        Require(nRays >= static_cast<std::uint64_t>(context.minValidRays),
                frameContext + ": n_rays below g2.min_valid_rays_per_frame");
        CompareCount(derived.nRays, static_cast<std::int64_t>(nRays), frameContext + " n_rays");
        Require(RequireCount(entry, "full_count", frameContext) ==
                    static_cast<std::int64_t>(nRays),
                frameContext + ": full_count disagrees with n_rays");
        Require(entry.contains("occluded_excluded_count"),
                frameContext + ": missing occluded_excluded_count");

        const nlohmann::json &buckets = RequireObject(entry, "bucket_counts", frameContext);
        for (std::size_t index = 0; index < kBucketKeys.size(); ++index)
        {
            CompareCount(derived.buckets[index],
                         RequireCount(buckets, kBucketKeys[index], frameContext),
                         frameContext + " bucket " + kBucketKeys[index]);
        }
        const nlohmann::json &cameras = RequireObject(entry, "camera_counts", frameContext);
        const std::int64_t image00 = RequireCount(cameras, "image_00", frameContext);
        const std::int64_t image01 = RequireCount(cameras, "image_01", frameContext);
        CompareCount(derived.cameras[0], image00, frameContext + " camera image_00");
        CompareCount(derived.cameras[1], image01, frameContext + " camera image_01");
        Require(derived.cameras[0] + derived.cameras[1] == derived.nRays,
                frameContext + ": every ray must have exactly one camera");
        if (context.requireBothCameras)
        {
            Require(derived.cameras[0] > 0U && derived.cameras[1] > 0U,
                    frameContext + ": both image_00 and image_01 must have rays");
        }
        CompareCount(derived.occluded,
                     RequireCount(entry, "occluded_count", frameContext),
                     frameContext + " occluded_count");
        CompareCount(derived.unoccludedPrimary,
                     RequireCount(entry, "occluded_excluded_count", frameContext),
                     frameContext + " occluded_excluded_count");

        const fs::path directory = manifestPath.parent_path();
        const std::string primaryName = FileNameFor(entry, "primary_mask", frameContext);
        const std::string secondaryName = FileNameFor(entry, "secondary_mask", frameContext);
        const std::string unoccludedName =
            FileNameFor(entry, "primary_unoccluded_mask", frameContext);
        Require(primaryName != secondaryName,
                frameContext + ": primary_mask and secondary_mask must be distinct files");

        const std::vector<std::uint8_t> primary = ReadBytes(directory / primaryName);
        const std::vector<std::uint8_t> secondary = ReadBytes(directory / secondaryName);
        const std::vector<std::uint8_t> unoccluded = ReadBytes(directory / unoccludedName);
        for (const auto *mask : {&primary, &secondary, &unoccluded})
        {
            Require(mask->size() == derived.nRays,
                    frameContext + ": mask length does not equal n_rays");
            for (const std::uint8_t value : *mask)
            {
                Require(value <= 1U, frameContext + ": mask value is not 0/1");
            }
        }
        const auto ones = [](const std::vector<std::uint8_t> &mask) {
            return static_cast<std::uint64_t>(
                std::count(mask.begin(), mask.end(), static_cast<std::uint8_t>(1U)));
        };
        Require(ones(primary) == derived.nRays - derived.buckets[3],
                frameContext + ": primary_mask count disagrees with the rays");
        Require(ones(secondary) == derived.buckets[3],
                frameContext + ": secondary_mask count disagrees with the rays");
        Require(ones(unoccluded) == derived.unoccludedPrimary,
                frameContext + ": primary_unoccluded_mask count disagrees with the rays");

        const nlohmann::json meta =
            ReadJson(directory / ("frame_" + FrameName(frame) + "_meta.json"));
        CompareCount(derived.nRays, RequireCount(meta, "n_rays", frameContext + " meta"),
                     frameContext + " meta n_rays");
        const nlohmann::json &metaBuckets = RequireObject(meta, "bucket_counts", frameContext);
        for (std::size_t index = 0; index < kBucketKeys.size(); ++index)
        {
            CompareCount(derived.buckets[index],
                         RequireCount(metaBuckets, kBucketKeys[index], frameContext),
                         frameContext + " meta bucket " + kBucketKeys[index]);
        }
        const nlohmann::json &metaCameras = RequireObject(meta, "camera_counts", frameContext);
        CompareCount(derived.cameras[0], RequireCount(metaCameras, "image_00", frameContext),
                     frameContext + " meta image_00");
        CompareCount(derived.cameras[1], RequireCount(metaCameras, "image_01", frameContext),
                     frameContext + " meta image_01");
        CompareCount(derived.occluded, RequireCount(meta, "occluded_count", frameContext),
                     frameContext + " meta occluded_count");
        CompareCount(derived.unoccludedPrimary,
                     RequireCount(meta, "occluded_excluded_count", frameContext),
                     frameContext + " meta occluded_excluded_count");

        totalRays += nRays;
        minRays = std::min(minRays, nRays);
    }
    RequireSameFrames(rayFrames, frames, manifestPath.string());

    return {{"frames", entries.size()},
            {"total_rays", totalRays},
            {"min_rays_per_frame", minRays},
            {"min_valid_rays_setting", context.minValidRays},
            {"require_image_00_and_01", context.requireBothCameras}};
}

nlohmann::json CheckHash(const Context &context)
{
    const std::vector<int> frames = RequireSelected(context);
    const fs::path manifestPath = context.phaseDir / "rays" / "rays_manifest.json";
    const fs::path freezePath = context.phaseDir / "rays" / "RAY_SET_FREEZE.json";

    const nlohmann::json result = rays::VerifyRaySet(manifestPath, freezePath);
    Require(result.at("ok").get<bool>(), "VerifyRaySet reported a mismatch");

    const nlohmann::json freeze = ReadJson(freezePath);
    const std::string freezeContext = freezePath.string();
    const nlohmann::json &perFrame = RequireObject(freeze, "per_frame_sha256", freezeContext);
    const nlohmann::json &perMeta =
        RequireObject(freeze, "per_frame_meta_sha256", freezeContext);
    for (const int frame : frames)
    {
        const std::string key = std::to_string(frame);
        Require(perFrame.contains(key) && perFrame.at(key).is_string(),
                freezePath.string() + ": missing per_frame_sha256 for " + key);
        Require(perMeta.contains(key) && perMeta.at(key).is_string(),
                freezePath.string() + ": missing per_frame_meta_sha256 for " + key);
    }
    Require(RequireString(freeze, "manifest_sha256", freezePath.string()) ==
                Sha256File(manifestPath),
            freezePath.string() + ": manifest_sha256 mismatch");
    Require(RequireString(freeze, "config_sha256", freezePath.string()) ==
                Sha256File(context.configPath),
            freezePath.string() + ": config_sha256 mismatch");
    const std::string raySetSha = RequireString(freeze, "ray_set_sha256", freezePath.string());
    Require(!raySetSha.empty(), freezePath.string() + ": ray_set_sha256 is empty");

    const std::vector<int> window =
        IntArray(RequireArray(freeze, "window", freezePath.string()), "freeze window");
    RequireSameFrames(window, frames, freezePath.string());

    return {{"verified", true},
            {"ray_set_sha256", raySetSha},
            {"manifest_sha256", freeze.at("manifest_sha256")},
            {"frames", frames.size()}};
}

nlohmann::json CheckWaterfall(const Context &context)
{
    const std::vector<int> frames = RequireSelected(context);
    const fs::path path = context.phaseDir / "rays" / "waterfalls.json";
    const nlohmann::json document = ReadJson(path);
    const std::string documentContext = path.string();
    const std::vector<int> window =
        IntArray(RequireArray(document, "window", documentContext), "waterfall window");
    RequireSameFrames(window, frames, documentContext);
    const nlohmann::json &entries = RequireArray(document, "frames", documentContext);
    Require(entries.size() == static_cast<std::size_t>(kWindowFrames),
            documentContext + ": must contain exactly 60 frames");

    std::vector<int> waterfallFrames;
    std::int64_t totalRaw = 0;
    std::int64_t totalValid = 0;
    for (const nlohmann::json &entry : entries)
    {
        const int frame = static_cast<int>(RequireInt(entry, "frame", path.string()));
        waterfallFrames.push_back(frame);
        rays::FrameWaterfall waterfall;
        waterfall.frame = frame;
        waterfall.raw = RequireCount(entry, "raw", path.string());
        waterfall.nonfinite = RequireCount(entry, "nonfinite", path.string());
        waterfall.ego_roi = RequireCount(entry, "ego_roi", path.string());
        waterfall.dynamic_union = RequireCount(entry, "dynamic_union", path.string());
        waterfall.no_camera_fov = RequireCount(entry, "no_camera_fov", path.string());
        waterfall.range_out_of_domain = RequireCount(entry, "range_out_of_domain", path.string());
        waterfall.valid = RequireCount(entry, "valid", path.string());
        std::string reason;
        Require(rays::ValidateWaterfall(waterfall, &reason),
                path.string() + " frame " + std::to_string(frame) + ": " + reason);
        Require(GetBool(entry, "exact", false),
                path.string() + " frame " + std::to_string(frame) + ": exact is not true");
        totalRaw += waterfall.raw;
        totalValid += waterfall.valid;
    }
    RequireSameFrames(waterfallFrames, frames, path.string());

    const nlohmann::json &totals = RequireObject(document, "totals", documentContext);
    Require(GetBool(totals, "exact", false), path.string() + ": totals.exact is not true");
    Require(RequireCount(totals, "raw", path.string()) == totalRaw,
            path.string() + ": totals.raw disagrees with the frames");
    Require(RequireCount(totals, "valid", path.string()) == totalValid,
            path.string() + ": totals.valid disagrees with the frames");

    return {{"frames", entries.size()}, {"raw", totalRaw}, {"valid", totalValid}};
}

nlohmann::json CheckDomain(const Context &context)
{
    const std::vector<int> frames = RequireSelected(context);
    const nlohmann::json manifest =
        ReadJson(context.phaseDir / "rays" / "rays_manifest.json");
    const auto entries = FrameIndex(manifest, "rays_manifest.json");

    const calib::Kitti360Calib calibration =
        calib::LoadKitti360Calib(context.datasetRoot / "calibration");
    const std::map<int, Eigen::Matrix4d> poses =
        calib::LoadPoses(context.datasetRoot / "data_poses" / context.sequence / "poses.txt");

    std::uint64_t sampled = 0;
    std::uint64_t allRays = 0;
    for (const int frame : frames)
    {
        const auto entry = entries.find(frame);
        Require(entry != entries.end(),
                "rays_manifest.json: missing frame " + std::to_string(frame));
        const fs::path directory = context.phaseDir / "rays";
        const std::string rayName = FileNameFor(*entry->second, "ray", "rays domain");
        int headerFrame = -1;
        const std::vector<rays::Ray> rays =
            rays::ReadRayFrame(directory / rayName, &headerFrame);
        Require(headerFrame == frame, "rays domain: header frame mismatch");

        const auto pose = poses.find(frame);
        Require(pose != poses.end(),
                "rays domain: frame " + std::to_string(frame) + " has no exact pose");

        std::map<std::string, Eigen::Matrix4d> transforms;
        std::map<std::string, Eigen::Matrix3d> intrinsics;
        std::map<std::string, std::pair<int, int>> imageSizes;
        for (const std::string &camera : context.cameras)
        {
            transforms[camera] = calib::ComposeTCkRectV(calibration, pose->second, camera);
            const auto k = calibration.K_rect.find(camera);
            Require(k != calibration.K_rect.end(), "rays domain: missing K_rect for " + camera);
            intrinsics[camera] = k->second;
            const fs::path imagePath = context.phaseDir / "masks" /
                                       ("frame_" + FrameName(frame)) /
                                       (camera + "_dynamic.png");
            int width = 0;
            int height = 0;
            int channels = 0;
            Require(stbi_info(imagePath.string().c_str(), &width, &height, &channels) != 0,
                    "rays domain: cannot read image size from " + imagePath.string());
            Require(width > 0 && height > 0,
                    "rays domain: invalid image size in " + imagePath.string());
            imageSizes[camera] = {width, height};
        }
        Require(context.cameras.size() >= 2U,
                "rays domain: both image_00 and image_01 are required");

        const std::size_t step =
            std::max<std::size_t>(1, rays.size() / kDomainSamplePerFrame);
        for (std::size_t index = 0; index < rays.size(); ++index)
        {
            const rays::Ray &ray = rays[index];
            allRays += 1U;
            Require(ray.t_gt >= context.rMinM && ray.t_gt <= context.rMaxM,
                    "rays domain: t_gt outside [1.5, 60]");
            Require(static_cast<double>(ray.t_gt) * static_cast<double>(ray.dir_C.z()) > 0.0,
                    "rays domain: z_cam is not positive");
            Require(std::abs(static_cast<double>(ray.dir_C.norm()) - 1.0) <=
                        kDirectionNormTolerance,
                    "rays domain: direction is not unit length");
            Require(ray.camera < 2U, "rays domain: camera is not 0 or 1");
            Require(ray.bucket < 4U, "rays domain: bucket is outside 0-3");
            if (index % step != 0U)
            {
                continue;
            }

            const std::string &camera = context.cameras[ray.camera];
            const Eigen::Vector4d pL(static_cast<double>(ray.p_L.x()),
                                     static_cast<double>(ray.p_L.y()),
                                     static_cast<double>(ray.p_L.z()), 1.0);
            const Eigen::Vector4d pC = transforms.at(camera) * pL;
            Require(std::abs(pC.x() - static_cast<double>(ray.p_C.x())) <=
                        kPoseReprojectionTolerance &&
                        std::abs(pC.y() - static_cast<double>(ray.p_C.y())) <=
                            kPoseReprojectionTolerance &&
                        std::abs(pC.z() - static_cast<double>(ray.p_C.z())) <=
                            kPoseReprojectionTolerance,
                    "rays domain: p_C does not match T_Ck_rect_V * p_L");
            Require(pC.z() > 0.0, "rays domain: z_cam <= 0 after recomposition");

            const Eigen::Vector3d projected = intrinsics.at(camera) * pC.head<3>();
            Require(projected.z() > 0.0, "rays domain: projected depth is not positive");
            const double u = projected.x() / projected.z();
            const double v = projected.y() / projected.z();
            const auto size = imageSizes.at(camera);
            Require(static_cast<double>(context.marginPx) <= u &&
                        u <= static_cast<double>(size.first - 1 - context.marginPx) &&
                        static_cast<double>(context.marginPx) <= v &&
                        v <= static_cast<double>(size.second - 1 - context.marginPx),
                    "rays domain: projection violates the 5 px image margin");

            const double rGt = pC.head<3>().norm();
            Require(std::abs(rGt - static_cast<double>(ray.t_gt)) <=
                        kRangeReprojectionTolerance,
                    "rays domain: r_gt does not equal ||p_C||");
            sampled += 1U;
        }
    }

    return {{"frames", frames.size()},
            {"rays_checked", allRays},
            {"rays_reprojected", sampled},
            {"r_gt_definition", "camera-frame Euclidean ||p_C||"}};
}

void AddCheck(nlohmann::json &checks, const std::string &id,
              const std::function<nlohmann::json()> &body)
{
    try
    {
        checks.push_back({{"id", id}, {"status", "pass"}, {"evidence", body()}});
    }
    catch (const std::exception &error)
    {
        checks.push_back({{"id", id}, {"status", "fail"}, {"evidence", {{"error", error.what()}}}});
    }
}
} // namespace

nlohmann::json EvaluateG2(const fs::path &phaseDir, const fs::path &configPath)
{
    Context context = LoadContext(phaseDir, configPath);

    nlohmann::json report;
    report["schema_version"] = 1;
    report["generated_at_utc"] = UtcNowIso8601();
    report["sequence"] = context.sequence;
    report["run_id"] = "";
    report["decisions_sha256"] = "";
    report["protocol_sha256"] = "";
    report["frozen_at_utc"] = "";
    report["protocol"] = {{"r_gt_definition", "camera-frame Euclidean ||p_C||"},
                          {"occlusion_source", "GT_only"},
                          {"lidar_origin_beam_protocol", "not_in_phase_2"}};
    report["limitations"] = nlohmann::json::array(
        {"no lateral-error claim; GT-defined rays have zero constructed lateral error",
         "LiDAR-origin beam-pattern secondary protocol is deferred (D-23)"});
    report["checks"] = nlohmann::json::array();

    AddCheck(report["checks"], "prereg", [&]() {
        return CheckPrereg(context, report);
    });

    AddCheck(report["checks"], "segment", [&]() {
        const nlohmann::json evidence = CheckSegment(context);
        context.selectedFrames = evidence.at("frames").get<std::vector<int>>();
        return evidence;
    });

    AddCheck(report["checks"], "uncurl", [&]() { return CheckUncurl(context); });
    AddCheck(report["checks"], "masks", [&]() { return CheckMasks(context); });
    AddCheck(report["checks"], "gt", [&]() { return CheckGt(context); });
    AddCheck(report["checks"], "rays", [&]() { return CheckRays(context); });
    AddCheck(report["checks"], "waterfall", [&]() { return CheckWaterfall(context); });
    AddCheck(report["checks"], "hash", [&]() { return CheckHash(context); });
    AddCheck(report["checks"], "domain", [&]() { return CheckDomain(context); });

    bool pass = true;
    std::size_t index = 0;
    for (const nlohmann::json &check : report["checks"])
    {
        Require(index < kCheckIds.size() && check.at("id").get<std::string>() == kCheckIds[index],
                "internal: unexpected G2 check order");
        ++index;
        if (check.at("status").get<std::string>() != "pass")
        {
            pass = false;
        }
    }
    Require(index == kCheckIds.size(), "internal: a G2 check is missing");
    report["pass"] = pass;
    return report;
}

std::string RenderG2Markdown(const nlohmann::json &report)
{
    std::ostringstream out;
    out << "# G2 Report\n\n";
    out << "- **Sequence:** " << report.value("sequence", std::string{}) << "\n";
    out << "- **Run ID:** " << report.value("run_id", std::string{}) << "\n";
    out << "- **Generated at:** " << report.value("generated_at_utc", std::string{}) << "\n";
    out << "- **Decisions SHA-256:** " << report.value("decisions_sha256", std::string{})
        << "\n";
    out << "- **Protocol SHA-256:** " << report.value("protocol_sha256", std::string{}) << "\n";
    out << "- **Frozen at:** " << report.value("frozen_at_utc", std::string{}) << "\n\n";
    out << "| Check | Status | Evidence |\n";
    out << "|---|---|---|\n";
    if (report.contains("checks"))
    {
        for (const nlohmann::json &check : report["checks"])
        {
            std::string evidence = check.value("evidence", nlohmann::json::object()).dump();
            if (evidence.size() > 240U)
            {
                evidence.resize(240U);
                evidence += "...";
            }
            out << "| " << check.value("id", std::string{}) << " | "
                << check.value("status", std::string{}) << " | `" << evidence << "` |\n";
        }
    }
    if (report.contains("protocol") && report["protocol"].is_object())
    {
        out << "\n**Protocol:** r_gt = "
            << report["protocol"].value("r_gt_definition", std::string{})
            << "; occlusion = " << report["protocol"].value("occlusion_source", std::string{})
            << "; LiDAR-origin beam protocol = "
            << report["protocol"].value("lidar_origin_beam_protocol", std::string{}) << ".\n";
    }
    if (report.contains("limitations") && report["limitations"].is_array())
    {
        out << "\n**Limitations:**\n";
        for (const nlohmann::json &limitation : report["limitations"])
        {
            out << "- " << limitation.get<std::string>() << "\n";
        }
    }
    out << "\n" << (report.value("pass", false) ? "G2 PASSED" : "G2 NOT PASSED") << "\n";
    return out.str();
}
} // namespace raylidar::g2
