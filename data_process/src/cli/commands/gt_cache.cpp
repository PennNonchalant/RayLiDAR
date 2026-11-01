#include <raylidar/calib/kitti360.h>
#include <raylidar/cli/command.h>
#include <raylidar/cli/commands_phase2.h>
#include <raylidar/common/artifacts.h>
#include <raylidar/common/cli_options.h>
#include <raylidar/common/io.h>
#include <raylidar/common/mask_bits.h>
#include <raylidar/gt/cache.h>
#include <raylidar/gt/manifest.h>
#include <raylidar/gt/camera_assign.h>
#include <raylidar/lidar/uncurl.h>
#include <raylidar/lidar/velodyne.h>
#include <raylidar/masking/semantics.h>

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
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
using raylidar::ParsedArgs;
using raylidar::ParseArgs;
using raylidar::PhaseArtifactsDir;
using raylidar::UsageError;
using raylidar::calib::ComposeTCkRectV;
using raylidar::calib::ComposeTWV;
using raylidar::calib::Kitti360Calib;
using raylidar::calib::LoadKitti360Calib;
using raylidar::calib::LoadPoses;
using raylidar::gt::AssignPrimaryCamera;
using raylidar::gt::BuildGtManifest;
using raylidar::gt::CameraAssignment;
using raylidar::gt::GtPoint;
using raylidar::gt::WriteGtFrame;
using raylidar::lidar::LoadVelodyneScan;
using raylidar::lidar::LoadVelodyneTimestamps;
using raylidar::lidar::PointXYZI;
using raylidar::lidar::UncurlParams;
using raylidar::lidar::UncurlScan;
using raylidar::masking::IsKeptStatic;
using raylidar::masking::LabelAt;
using raylidar::masking::LoadSemantics;
using raylidar::masking::SemanticImage;

constexpr int kExitInvalidInput = 4;
constexpr int kExitGate = 6;
constexpr std::uint16_t kUnknownLabel = 65535U;
constexpr std::uint8_t kUnknownCamera = 255U;

void PrintHelp(std::ostream &out)
{
    out << "Usage: raylidar_cli gt-cache [options]\n"
        << "\n"
        << "Build the hashed per-frame static GT cache for the frozen window.\n"
        << "Each kept-static point is uncurled, assigned to at most one primary\n"
        << "camera, given a camera-frame range and semantic label, and written to\n"
        << "gt/frame_<frame>.bin plus gt/gt_manifest.json.\n"
        << "\n"
        << "Options:\n"
        << "  --config <path>            Phase-2 YAML (default: configs/phase2.yaml)\n"
        << "  --segment-manifest <path>  Full segment manifest\n"
        << "                             (default: <phase_dir>/segment/segment_manifest.json)\n"
        << "  --gt-manifest <path>       Manifest file inside a gt/ directory\n"
        << "                             (default: <phase_dir>/gt/gt_manifest.json)\n"
        << "  --frames <list>            Comma-separated smoke frame ids\n"
        << "  --out-dir <path>           GT directory (default: <phase_dir>/gt)\n"
        << "  --help                     Show this help\n"
        << "\n"
        << "Exit codes: 0 success, 2 usage, 4 invalid input, 6 count gate failed.\n";
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
    Require(node && node.IsSequence() && !node.IsNull() && node.size() > 0U,
            source + " must be a non-empty sequence", kExitInvalidInput);
    std::vector<std::string> values;
    values.reserve(node.size());
    for (const YAML::Node &value : node)
    {
        Require(value.IsScalar(), source + " entries must be scalars", kExitInvalidInput);
        const std::string text = value.as<std::string>();
        Require(!text.empty(), source + " entries must not be empty", kExitInvalidInput);
        values.push_back(text);
    }
    return values;
}

std::vector<double> RequireRange(const YAML::Node &node, const std::string &source)
{
    Require(node && node.IsSequence() && node.size() == 2U,
            source + " must be a two-element range", kExitInvalidInput);
    const double lower = node[0].as<double>();
    const double upper = node[1].as<double>();
    Require(lower < upper, source + " lower bound must be below the upper bound",
            kExitInvalidInput);
    return {lower, upper};
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
        const std::string token =
            text.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        if (!token.empty())
        {
            try
            {
                std::size_t consumed = 0;
                const int frame = std::stoi(token, &consumed);
                Require(consumed == token.size(), source + ": invalid frame id '" + token + "'",
                        kExitInvalidInput);
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

/// Read the frozen 60-frame window from a full segment manifest.
///
/// Requires `selection_mode == "full"`, an integer `frame_count`, a
/// `selected.frames` array of exactly that length, and a strictly ascending
/// frame list. Any deviation is exit 4; a smoke/missing manifest never
/// produces a real GT cache.
std::vector<int> FramesFromSegmentManifest(const fs::path &path)
{
    const nlohmann::json document = raylidar::ReadJson(path);
    const std::string mode = document.value("selection_mode", "");
    Require(mode == "full",
            path.string() + ": selection_mode must be \"full\" for a real gt-cache run",
            kExitInvalidInput);
    Require(document.contains("selected") && document["selected"].is_object(),
            path.string() + ": missing selected object", kExitInvalidInput);

    const nlohmann::json &selected = document["selected"];
    Require(selected.contains("frame_count") && selected["frame_count"].is_number_integer(),
            path.string() + ": selected.frame_count must be an integer", kExitInvalidInput);
    const int frameCount = selected["frame_count"].get<int>();
    Require(frameCount > 0, path.string() + ": selected.frame_count must be positive",
            kExitInvalidInput);

    Require(selected.contains("frames") && selected["frames"].is_array(),
            path.string() + ": selected.frames must be an array", kExitInvalidInput);
    const nlohmann::json &array = selected["frames"];
    Require(array.size() == static_cast<std::size_t>(frameCount),
            path.string() + ": selected.frames length does not match selected.frame_count",
            kExitInvalidInput);

    std::vector<int> frames;
    frames.reserve(array.size());
    for (const nlohmann::json &value : array)
    {
        Require(value.is_number_integer(),
                path.string() + ": selected frame ids must be integers", kExitInvalidInput);
        frames.push_back(value.get<int>());
    }
    for (std::size_t index = 1; index < frames.size(); ++index)
    {
        Require(frames[index] > frames[index - 1U],
                path.string() + ": selected.frames must be strictly ascending", kExitInvalidInput);
    }
    return frames;
}

std::vector<std::uint8_t> ReadBytesOrFail(const fs::path &path)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input.good())
    {
        throw IoError("cannot read mask points " + path.string(), kExitInvalidInput);
    }
    const std::streamoff end = input.tellg();
    Require(end >= 0, "cannot size mask points " + path.string(), kExitInvalidInput);
    input.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(end));
    if (!bytes.empty())
    {
        input.read(reinterpret_cast<char *>(bytes.data()),
                   static_cast<std::streamsize>(bytes.size()));
        Require(input.good(), "cannot read mask points " + path.string(), kExitInvalidInput);
    }
    return bytes;
}

int Execute(std::span<const std::string_view> args)
{
    const ParsedArgs parsed = ParseArgs(
        args, {"config", "segment-manifest", "gt-manifest", "frames", "out-dir"}, {"help"});
    if (parsed.Flag("help"))
    {
        PrintHelp(std::cout);
        return 0;
    }

    const std::string configPath = parsed.Get("config", "configs/phase2.yaml");
    const YAML::Node config = raylidar::ReadYaml(configPath);
    Require(config && config.IsMap(), configPath + ": expected a YAML mapping", kExitInvalidInput);

    const std::string datasetRoot = NodeString(config, "dataset_root", "");
    const std::string lidarRoot = NodeString(config, "lidar_root", "");
    const std::string sequence = NodeString(config, "sequence", "");
    Require(!datasetRoot.empty(), configPath + ": dataset_root is not configured",
            kExitInvalidInput);
    Require(!lidarRoot.empty(), configPath + ": lidar_root is not configured", kExitInvalidInput);
    Require(!sequence.empty(), configPath + ": sequence is not configured", kExitInvalidInput);

    const std::vector<std::string> cameras =
        NodeStrings(config["cameras"], configPath + ": cameras");
    Require(cameras.size() >= 2U && cameras[0] == "image_00" && cameras[1] == "image_01",
            configPath + ": cameras must start with image_00 and image_01", kExitInvalidInput);

    fs::path phaseDir = NodeString(config["artifacts"], "phase_dir", "");
    if (phaseDir.empty())
    {
        phaseDir = PhaseArtifactsDir(2);
    }

    const YAML::Node gtConfig = config["gt_cache"];
    Require(gtConfig && gtConfig.IsMap(), configPath + ": missing gt_cache section",
            kExitInvalidInput);
    Require(NodeString(gtConfig, "endianness", "") == "little",
            configPath + ": gt_cache.endianness must be little", kExitInvalidInput);
    Require(NodeInt(gtConfig, "record_size", -1) == 32,
            configPath + ": gt_cache.record_size must be 32", kExitInvalidInput);
    const int minPointsPerFrame = NodeInt(gtConfig, "min_points_per_frame", -1);
    Require(minPointsPerFrame >= 0, configPath + ": gt_cache.min_points_per_frame must be set",
            kExitInvalidInput);
    const std::vector<double> expectedRange =
        RequireRange(gtConfig["expected_points_range"], configPath + ": gt_cache.expected_points_range");
    Require(NodeInt(gtConfig, "camera_unknown", -1) == static_cast<int>(kUnknownCamera),
            configPath + ": gt_cache.camera_unknown must be 255", kExitInvalidInput);

    const YAML::Node rays = config["rays"];
    Require(rays && rays.IsMap(), configPath + ": missing rays section", kExitInvalidInput);
    const double rMinM = NodeDouble(rays, "r_min_m", -1.0);
    const double rMaxM = NodeDouble(rays, "r_max_m", -1.0);
    const int marginPx = NodeInt(rays, "image_margin_px", -1);
    Require(rMinM > 0.0, configPath + ": rays.r_min_m must be positive", kExitInvalidInput);
    Require(rMaxM > rMinM, configPath + ": rays.r_max_m must exceed rays.r_min_m",
            kExitInvalidInput);
    Require(marginPx >= 0, configPath + ": rays.image_margin_px must be non-negative",
            kExitInvalidInput);

    fs::path outDir = parsed.Get("out-dir", "");
    if (outDir.empty())
    {
        outDir = phaseDir / "gt";
    }

    fs::path manifestPath = parsed.Get("gt-manifest", "");
    if (manifestPath.empty())
    {
        manifestPath = phaseDir / "gt" / "gt_manifest.json";
    }
    Require(manifestPath.parent_path().filename() == "gt",
            "--gt-manifest must live inside a 'gt' directory", kExitInvalidInput);

    const std::string framesText = parsed.Get("frames", "");
    const bool smoke = !framesText.empty();
    std::vector<int> frames;
    if (smoke)
    {
        frames = ParseFrameList(framesText, "--frames");
        manifestPath = manifestPath.parent_path() / "gt_manifest.smoke.json";
    }
    else
    {
        fs::path segmentManifest = parsed.Get("segment-manifest", "");
        if (segmentManifest.empty())
        {
            segmentManifest = phaseDir / "segment" / "segment_manifest.json";
        }
        frames = FramesFromSegmentManifest(segmentManifest);
    }
    std::sort(frames.begin(), frames.end());
    frames.erase(std::unique(frames.begin(), frames.end()), frames.end());
    Require(!frames.empty(), "gt-cache needs at least one frame", kExitInvalidInput);

    const Kitti360Calib calib = LoadKitti360Calib(fs::path(datasetRoot) / "calibration");
    for (const std::string &camera : cameras)
    {
        Require(calib.K_rect.find(camera) != calib.K_rect.end(),
                "calibration has no rectified K for " + camera, kExitInvalidInput);
    }
    const std::map<int, Eigen::Matrix4d> poses =
        LoadPoses(fs::path(datasetRoot) / "data_poses" / sequence / "poses.txt");
    const std::vector<std::int64_t> timestamps =
        LoadVelodyneTimestamps(fs::path(lidarRoot) / "timestamps.txt");

    raylidar::EnsureDir(outDir);
    const std::string configSha = raylidar::Sha256File(configPath);

    std::vector<nlohmann::json> frameDocuments;
    frameDocuments.reserve(frames.size());

    for (const int frame : frames)
    {
        const auto currentPose = poses.find(frame);
        const auto nextPose = poses.find(frame + 1);
        Require(currentPose != poses.end(),
                "frame " + std::to_string(frame) + " has no exact pose", kExitInvalidInput);
        Require(nextPose != poses.end(),
                "frame " + std::to_string(frame) + " has no exact successor pose (uncurl)",
                kExitInvalidInput);
        Require(frame >= 0 && static_cast<std::size_t>(frame + 1) < timestamps.size(),
                "frame " + std::to_string(frame) +
                    " has no timestamp for itself and its successor (uncurl)",
                kExitInvalidInput);

        const fs::path scanPath = fs::path(lidarRoot) / "data" / (FrameName(frame) + ".bin");
        const std::vector<PointXYZI> scan = LoadVelodyneScan(scanPath);

        UncurlParams params;
        params.T_W_V_curr = ComposeTWV(calib, currentPose->second);
        params.T_W_V_next = ComposeTWV(calib, nextPose->second);
        params.t_curr_ns = timestamps[static_cast<std::size_t>(frame)];
        params.t_next_ns = timestamps[static_cast<std::size_t>(frame + 1)];
        std::size_t rejected = 0;
        const std::vector<Eigen::Vector3d> finite = UncurlScan(scan, params, &rejected);
        std::vector<Eigen::Vector3d> uncurled(
            scan.size(), Eigen::Vector3d(std::numeric_limits<double>::quiet_NaN(),
                                         std::numeric_limits<double>::quiet_NaN(),
                                         std::numeric_limits<double>::quiet_NaN()));
        std::size_t cursor = 0;
        for (std::size_t index = 0; index < scan.size(); ++index)
        {
            const Eigen::Vector3d raw(scan[index].x, scan[index].y, scan[index].z);
            if (raw.allFinite())
            {
                Require(cursor < finite.size(), "uncurl output is shorter than the finite input",
                        kExitInvalidInput);
                uncurled[index] = finite[cursor++];
            }
        }
        Require(cursor == finite.size(), "uncurl output does not match the scan",
                kExitInvalidInput);

        const fs::path maskPath =
            phaseDir / "masks" / ("frame_" + FrameName(frame)) / "points.bin";
        const std::vector<std::uint8_t> bits = ReadBytesOrFail(maskPath);
        Require(bits.size() == scan.size(),
                "mask point count does not match the raw scan for frame " +
                    std::to_string(frame),
                kExitInvalidInput);

        std::map<std::string, SemanticImage> semantics;
        std::map<std::string, std::pair<int, int>> imageSizes;
        std::map<std::string, Eigen::Matrix4d> cameraTransforms;
        std::map<std::string, Eigen::Matrix3d> cameraIntrinsics;
        for (const std::string &camera : cameras)
        {
            const fs::path semanticsPath = fs::path(datasetRoot) / "data_2d_semantics" / "train" /
                                           sequence / camera / "semantic" /
                                           (FrameName(frame) + ".png");
            SemanticImage image = LoadSemantics(semanticsPath);
            imageSizes[camera] = {image.width, image.height};
            semantics.emplace(camera, std::move(image));
            cameraTransforms[camera] = ComposeTCkRectV(calib, currentPose->second, camera);
            cameraIntrinsics[camera] = calib.K_rect.at(camera);
        }

        std::vector<GtPoint> points;
        points.reserve(scan.size());
        std::size_t camera0 = 0;
        std::size_t camera1 = 0;
        std::size_t cameraUnknown = 0;

        for (std::size_t index = 0; index < scan.size(); ++index)
        {
            if (!IsKeptStatic(bits[index]))
            {
                continue;
            }
            Require(uncurled[index].allFinite(),
                    "kept-static point is not finite for frame " + std::to_string(frame),
                    kExitInvalidInput);

            const CameraAssignment assignment =
                AssignPrimaryCamera(uncurled[index], cameraTransforms, cameraIntrinsics,
                                    imageSizes, rMinM, rMaxM, marginPx);
            std::uint16_t label = kUnknownLabel;
            if (assignment.camera != kUnknownCamera &&
                static_cast<std::size_t>(assignment.camera) < cameras.size())
            {
                const std::string &camera = cameras[static_cast<std::size_t>(assignment.camera)];
                label = LabelAt(semantics.at(camera), assignment.u, assignment.v);
            }

            GtPoint point;
            point.p_L = uncurled[index].cast<float>();
            point.frame = static_cast<std::uint32_t>(frame);
            point.semanticId = label;
            point.camera = assignment.camera;
            point.valid = 1U;
            point.r_gt = static_cast<float>(assignment.r_gt);
            points.push_back(point);

            if (assignment.camera == 0U)
            {
                ++camera0;
            }
            else if (assignment.camera == 1U)
            {
                ++camera1;
            }
            else
            {
                ++cameraUnknown;
            }
        }

        const std::size_t count = points.size();
        if (!smoke)
        {
            Require(count >= static_cast<std::size_t>(minPointsPerFrame),
                    "frame " + std::to_string(frame) + " has only " + std::to_string(count) +
                        " GT points; the per-frame minimum is " +
                        std::to_string(minPointsPerFrame),
                    kExitGate);
            Require(static_cast<double>(count) >= expectedRange[0] &&
                        static_cast<double>(count) <= expectedRange[1],
                    "frame " + std::to_string(frame) + " has " + std::to_string(count) +
                        " GT points outside [" + std::to_string(expectedRange[0]) + ", " +
                        std::to_string(expectedRange[1]) + "]",
                    kExitGate);
        }

        const fs::path framePath = outDir / ("frame_" + FrameName(frame) + ".bin");
        WriteGtFrame(framePath, frame, points);

        nlohmann::json entry;
        entry["frame"] = frame;
        entry["count"] = count;
        entry["camera_00"] = camera0;
        entry["camera_01"] = camera1;
        entry["camera_unknown"] = cameraUnknown;
        entry["sha256"] = raylidar::Sha256File(framePath);
        entry["config_sha256"] = configSha;
        frameDocuments.push_back(std::move(entry));

        std::cout << "frame=" << frame << " kept=" << count << " cam0=" << camera0
                  << " cam1=" << camera1 << " unknown=" << cameraUnknown << '\n';
    }

    BuildGtManifest(manifestPath.parent_path().parent_path(), sequence, frameDocuments,
                    manifestPath.filename().string());
    if (smoke)
    {
        std::cerr << "warning: smoke run wrote " << manifestPath.filename().string()
                  << "; this is not final\n";
    }
    return 0;
}
} // namespace

int RunGtCache(std::span<const std::string_view> args)
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

RAYLIDAR_REGISTER_COMMAND("gt-cache",
                          "Build the hashed per-frame static GT cache for the frozen window",
                          RunGtCache);
