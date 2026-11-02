#include <raylidar/calib/kitti360.h>
#include <raylidar/cli/command.h>
#include <raylidar/cli/commands_phase2.h>
#include <raylidar/common/artifacts.h>
#include <raylidar/common/cli_options.h>
#include <raylidar/common/io.h>
#include <raylidar/lidar/uncurl.h>
#include <raylidar/lidar/velodyne.h>
#include <raylidar/masking/bbox.h>
#include <raylidar/masking/mask.h>
#include <raylidar/masking/semantics.h>

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <limits>
#include <map>
#include <set>
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
using raylidar::calib::ComposeTWV;
using raylidar::calib::Kitti360Calib;
using raylidar::calib::LoadKitti360Calib;
using raylidar::calib::LoadPoses;
using raylidar::lidar::LoadVelodyneScan;
using raylidar::lidar::LoadVelodyneTimestamps;
using raylidar::lidar::PointXYZI;
using raylidar::lidar::UncurlParams;
using raylidar::lidar::UncurlScan;
using raylidar::masking::BuildMask;
using raylidar::masking::BuildMaskManifest;
using raylidar::masking::DynamicBBox;
using raylidar::masking::LoadDynamicBBoxes;
using raylidar::masking::LoadSemantics;
using raylidar::masking::MaskOptions;
using raylidar::masking::MaskResult;
using raylidar::masking::SemanticImage;
using raylidar::masking::WriteMaskArtifact;

constexpr int kExitInvalidInput = 4;

void PrintHelp(std::ostream &out)
{
    out << "Usage: raylidar_cli mask-build [options]\n"
        << "\n"
        << "Build the single dynamic/ego mask artifact consumed by images, GT,\n"
        << "projections, and evaluation. Writes masks/frame_<frame>/points.bin,\n"
        << "image_00_dynamic.png, image_01_dynamic.png, mask.json, and\n"
        << "masks/mask_manifest.json.\n"
        << "\n"
        << "Options:\n"
        << "  --config <path>       Phase-2 YAML (default: configs/phase2.yaml)\n"
        << "  --frames <list>       Comma-separated frame ids (wins over --frame-list)\n"
        << "  --frame-list <path>   JSON with usable_frames or selected.frames\n"
        << "                        (default: <phase_dir>/segment/segment_manifest.json)\n"
        << "  --out-dir <path>      Mask directory (default: <phase_dir>/masks)\n"
        << "  --max-frames <n>      Smoke cap; writes mask_manifest.smoke.json only\n"
        << "  --help                Show this help\n"
        << "\n"
        << "Exit codes: 0 success, 2 usage error, 4 invalid or missing input.\n";
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

void Require(const bool condition, const std::string &message)
{
    if (!condition)
    {
        throw IoError(message, kExitInvalidInput);
    }
}

std::string FrameName(const int frame)
{
    char buffer[16] = {};
    std::snprintf(buffer, sizeof(buffer), "%010d", frame);
    return std::string(buffer);
}

std::vector<double> RequireRange(const YAML::Node &node, const std::string &source)
{
    Require(node && node.IsSequence() && node.size() == 2U, source + " must be a two-element range");
    const double lower = node[0].as<double>();
    const double upper = node[1].as<double>();
    Require(lower < upper, source + " lower bound must be below the upper bound");
    return {lower, upper};
}

std::vector<std::string> NodeStrings(const YAML::Node &node, const std::string &source)
{
    Require(node && node.IsSequence() && !node.IsNull() && node.size() > 0U, source + " must be a non-empty sequence");
    std::vector<std::string> values;
    values.reserve(node.size());
    for (const YAML::Node &value : node)
    {
        Require(value.IsScalar(), source + " entries must be scalars");
        const std::string text = value.as<std::string>();
        Require(!text.empty(), source + " entries must not be empty");
        values.push_back(text);
    }
    return values;
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
                Require(consumed == token.size(), source + ": invalid frame id '" + token + "'");
                frames.push_back(frame);
            }
            catch (const std::invalid_argument &)
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
    Require(!frames.empty(), source + ": no frame ids supplied");
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
    Require(array != nullptr, path.string() + ": expected usable_frames or selected.frames");

    std::vector<int> frames;
    frames.reserve(array->size());
    for (const nlohmann::json &value : *array)
    {
        Require(value.is_number_integer(), path.string() + ": frame ids must be integers");
        frames.push_back(value.get<int>());
    }
    Require(!frames.empty(), path.string() + ": frame list is empty");
    return frames;
}

MaskOptions ReadMaskOptions(const YAML::Node &config, const std::string &configPath)
{
    const YAML::Node mask = config["mask"];
    Require(mask && mask.IsMap(), configPath + ": missing mask section");

    const std::set<int> expectedLabels = {24, 25, 26, 27, 28, 32, 33};
    const YAML::Node labelIds = mask["dynamic_label_ids"];
    Require(labelIds && labelIds.IsSequence(), configPath + ": missing mask.dynamic_label_ids");
    std::set<int> configured;
    for (const YAML::Node &value : labelIds)
    {
        configured.insert(value.as<int>());
    }
    Require(configured == expectedLabels,
            configPath + ": mask.dynamic_label_ids must match the frozen {24,25,26,27,28,32,33}");

    const YAML::Node roi = mask["ego_roi_velodyne_m"];
    Require(roi && roi.IsMap(), configPath + ": missing mask.ego_roi_velodyne_m");
    const std::vector<double> x = RequireRange(roi["x"], configPath + ": mask.ego_roi_velodyne_m.x");
    const std::vector<double> y = RequireRange(roi["y"], configPath + ": mask.ego_roi_velodyne_m.y");
    const std::vector<double> z = RequireRange(roi["z"], configPath + ": mask.ego_roi_velodyne_m.z");

    MaskOptions options;
    options.egoRoiXMin = x[0];
    options.egoRoiXMax = x[1];
    options.egoRoiYMin = y[0];
    options.egoRoiYMax = y[1];
    options.egoRoiZMin = z[0];
    options.egoRoiZMax = z[1];
    options.bboxMaxTimeGapFrames = NodeInt(mask, "bbox_max_time_gap_frames", -1);
    options.residualDynamicRadiusM = NodeDouble(mask, "residual_dynamic_radius_m", -1.0);
    Require(options.bboxMaxTimeGapFrames >= 0, configPath + ": mask.bbox_max_time_gap_frames must be non-negative");
    Require(options.residualDynamicRadiusM > 0.0, configPath + ": mask.residual_dynamic_radius_m must be positive");
    return options;
}

int Execute(std::span<const std::string_view> args)
{
    const ParsedArgs parsed = ParseArgs(args, {"config", "frames", "frame-list", "out-dir", "max-frames"}, {"help"});
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
    const std::string bboxesXml = NodeString(config, "bboxes_xml", "");
    const std::string sequence = NodeString(config, "sequence", "");
    Require(!datasetRoot.empty(), configPath + ": dataset_root is not configured");
    Require(!lidarRoot.empty(), configPath + ": lidar_root is not configured");
    Require(!bboxesXml.empty(), configPath + ": bboxes_xml is not configured");
    Require(!sequence.empty(), configPath + ": sequence is not configured");

    const std::vector<std::string> cameras = NodeStrings(config["cameras"], configPath + ": cameras");

    fs::path phaseDir = NodeString(config["artifacts"], "phase_dir", "");
    if (phaseDir.empty())
    {
        phaseDir = PhaseArtifactsDir(2);
    }

    const MaskOptions options = ReadMaskOptions(config, configPath);

    fs::path masksDir = parsed.Get("out-dir", "");
    if (masksDir.empty())
    {
        masksDir = phaseDir / "masks";
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

    bool smoke = false;
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
        smoke = true;
    }
    Require(!frames.empty(), "mask-build needs at least one frame");

    const Kitti360Calib calib = LoadKitti360Calib(fs::path(datasetRoot) / "calibration");
    const std::map<int, Eigen::Matrix4d> poses =
        LoadPoses(fs::path(datasetRoot) / "data_poses" / sequence / "poses.txt");
    const std::vector<std::int64_t> timestamps = LoadVelodyneTimestamps(fs::path(lidarRoot) / "timestamps.txt");
    const std::vector<DynamicBBox> boxes = LoadDynamicBBoxes(bboxesXml);

    std::vector<nlohmann::json> frameDocuments;
    frameDocuments.reserve(frames.size());
    for (const int frame : frames)
    {
        const auto currentPose = poses.find(frame);
        const auto nextPose = poses.find(frame + 1);
        if (currentPose == poses.end() || nextPose == poses.end())
        {
            throw IoError("frame " + std::to_string(frame) + " needs exact poses for itself and its successor (uncurl)",
                          kExitInvalidInput);
        }
        if (static_cast<std::size_t>(frame + 1) >= timestamps.size())
        {
            throw IoError("frame " + std::to_string(frame) + " needs timestamps for itself and its successor (uncurl)",
                          kExitInvalidInput);
        }

        const fs::path scanPath = fs::path(lidarRoot) / "data" / (FrameName(frame) + ".bin");
        const std::vector<PointXYZI> scan = LoadVelodyneScan(scanPath);

        UncurlParams params;
        params.T_W_V_curr = ComposeTWV(calib, currentPose->second);
        params.T_W_V_next = ComposeTWV(calib, nextPose->second);
        params.t_curr_ns = timestamps[static_cast<std::size_t>(frame)];
        params.t_next_ns = timestamps[static_cast<std::size_t>(frame + 1)];

        std::size_t rejected = 0;
        const std::vector<Eigen::Vector3d> finite = UncurlScan(scan, params, &rejected);
        std::vector<Eigen::Vector3d> uncurled(scan.size(), Eigen::Vector3d(std::numeric_limits<double>::quiet_NaN(),
                                                                           std::numeric_limits<double>::quiet_NaN(),
                                                                           std::numeric_limits<double>::quiet_NaN()));
        std::size_t cursor = 0;
        for (std::size_t index = 0; index < scan.size(); ++index)
        {
            const Eigen::Vector3d raw(scan[index].x, scan[index].y, scan[index].z);
            if (raw.allFinite())
            {
                if (cursor >= finite.size())
                {
                    throw IoError("uncurl output is shorter than the finite input", kExitInvalidInput);
                }
                uncurled[index] = finite[cursor++];
            }
        }
        if (cursor != finite.size())
        {
            throw IoError("uncurl output does not match the scan", kExitInvalidInput);
        }

        std::map<std::string, SemanticImage> semantics;
        std::map<std::string, std::pair<int, int>> imageSizes;
        std::map<std::string, Eigen::Matrix4d> cameraTransforms;
        for (const std::string &camera : cameras)
        {
            const fs::path semanticsPath = fs::path(datasetRoot) / "data_2d_semantics" / "train" / sequence / camera /
                                           "semantic" / (FrameName(frame) + ".png");
            SemanticImage image = LoadSemantics(semanticsPath);
            imageSizes[camera] = {image.width, image.height};
            semantics.emplace(camera, std::move(image));
            cameraTransforms[camera] = ComposeTCkRectV(calib, currentPose->second, camera);
        }

        const MaskResult result = BuildMask(scan, uncurled, frame, calib, params.T_W_V_curr, boxes, semantics,
                                            imageSizes, cameraTransforms, options);
        frameDocuments.push_back(WriteMaskArtifact(masksDir, frame, result, scan, sequence));

        std::cout << "frame=" << frame << " kept=" << result.counts["kept_static"].get<std::size_t>()
                  << " dynamic=" << result.counts["dynamic_union"].get<std::size_t>()
                  << " ego=" << result.counts["ego_roi"].get<std::size_t>() << '\n';
    }

    const std::string configSha = raylidar::Sha256File(configPath);
    BuildMaskManifest(masksDir, frameDocuments, sequence, configSha,
                      smoke ? "mask_manifest.smoke.json" : "mask_manifest.json");
    if (smoke)
    {
        std::cerr << "warning: smoke run wrote mask_manifest.smoke.json; this is not final\n";
    }
    return 0;
}
} // namespace

int RunMaskBuild(std::span<const std::string_view> args)
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

RAYLIDAR_REGISTER_COMMAND("mask-build", "Build the single dynamic/ego point+pixel mask artifact", RunMaskBuild);
