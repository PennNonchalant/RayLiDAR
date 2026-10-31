#include <raylidar/calib/kitti360.h>
#include <raylidar/cli/command.h>
#include <raylidar/cli/commands_phase2.h>
#include <raylidar/common/artifacts.h>
#include <raylidar/common/cli_options.h>
#include <raylidar/common/io.h>
#include <raylidar/lidar/velodyne.h>
#include <raylidar/masking/bbox.h>
#include <raylidar/segment/select.h>

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <span>
#include <sstream>
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
using raylidar::WriteJson;
using raylidar::calib::Kitti360Calib;
using raylidar::calib::LoadKitti360Calib;
using raylidar::calib::LoadPoses;
using raylidar::lidar::LoadVelodyneTimestamps;
using raylidar::masking::DynamicBBox;
using raylidar::masking::LoadDynamicBBoxes;
using raylidar::masking::MaskOptions;
using raylidar::segment::CandidateWindows;
using raylidar::segment::FrameScore;
using raylidar::segment::ScoreFrame;
using raylidar::segment::SelectionManifest;
using raylidar::segment::SelectWindow;
using raylidar::segment::SegmentInputs;
using raylidar::segment::WindowScore;

constexpr int kExitInvalidInput = 4;
constexpr int kExitNoCandidate = 6;

void PrintHelp(std::ostream &out)
{
    out << "Usage: raylidar_cli segment-select [options]\n"
        << "\n"
        << "Score every usable frame by the frozen dynamic union and select one\n"
        << "60-frame window with the lowest point-weighted dynamic fraction.\n"
        << "A full run writes segment/segment_manifest.json and\n"
        << "segment/candidate_windows.csv; a smoke run writes only\n"
        << "segment_manifest.smoke.json and never claims the final manifest.\n"
        << "\n"
        << "Options:\n"
        << "  --config <path>       Phase-2 YAML (default: configs/phase2.yaml)\n"
        << "  --frame-index <path>  Frame index JSON\n"
        << "                        (default: <phase_dir>/streams/frame_index.json)\n"
        << "  --bboxes <path>       Override bboxes_xml from the config\n"
        << "  --out-dir <path>      Segment directory (default: <phase_dir>/segment)\n"
        << "  --max-frames <n>      Smoke cap; implies a smoke run\n"
        << "  --stride <n>          Smoke stride; implies a smoke run\n"
        << "  --help                Show this help\n"
        << "\n"
        << "Exit codes: 0 success, 2 usage, 4 invalid input, 6 no candidate window.\n";
}

void Require(const bool condition, const std::string &message)
{
    if (!condition)
    {
        throw IoError(message, kExitInvalidInput);
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
            source + " must be a non-empty sequence");
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

std::vector<double> RequireRange(const YAML::Node &node, const std::string &source)
{
    Require(node && node.IsSequence() && node.size() == 2U,
            source + " must be a two-element range");
    const double lower = node[0].as<double>();
    const double upper = node[1].as<double>();
    Require(lower < upper, source + " lower bound must be below the upper bound");
    return {lower, upper};
}

MaskOptions ReadMaskOptions(const YAML::Node &config, const std::string &configPath)
{
    const YAML::Node mask = config["mask"];
    Require(mask && mask.IsMap(), configPath + ": missing mask section");

    const YAML::Node roi = mask["ego_roi_velodyne_m"];
    Require(roi && roi.IsMap(), configPath + ": missing mask.ego_roi_velodyne_m");
    const std::vector<double> x =
        RequireRange(roi["x"], configPath + ": mask.ego_roi_velodyne_m.x");
    const std::vector<double> y =
        RequireRange(roi["y"], configPath + ": mask.ego_roi_velodyne_m.y");
    const std::vector<double> z =
        RequireRange(roi["z"], configPath + ": mask.ego_roi_velodyne_m.z");

    MaskOptions options;
    options.egoRoiXMin = x[0];
    options.egoRoiXMax = x[1];
    options.egoRoiYMin = y[0];
    options.egoRoiYMax = y[1];
    options.egoRoiZMin = z[0];
    options.egoRoiZMax = z[1];
    options.bboxMaxTimeGapFrames = NodeInt(mask, "bbox_max_time_gap_frames", -1);
    options.residualDynamicRadiusM = NodeDouble(mask, "residual_dynamic_radius_m", -1.0);
    Require(options.bboxMaxTimeGapFrames >= 0,
            configPath + ": mask.bbox_max_time_gap_frames must be non-negative");
    Require(options.residualDynamicRadiusM > 0.0,
            configPath + ": mask.residual_dynamic_radius_m must be positive");
    return options;
}

std::vector<int> ReadUsableFrames(const fs::path &path)
{
    Require(fs::is_regular_file(path), "frame index not found: " + path.string());
    const nlohmann::json document = raylidar::ReadJson(path);
    Require(document.contains("usable_frames") && document["usable_frames"].is_array(),
            path.string() + ": expected a usable_frames array");
    std::vector<int> frames;
    frames.reserve(document["usable_frames"].size());
    for (const nlohmann::json &value : document["usable_frames"])
    {
        Require(value.is_number_integer(),
                path.string() + ": usable frame ids must be integers");
        frames.push_back(value.get<int>());
    }
    Require(!frames.empty(), path.string() + ": usable_frames is empty");
    std::sort(frames.begin(), frames.end());
    frames.erase(std::unique(frames.begin(), frames.end()), frames.end());
    return frames;
}

int ParsePositive(const std::string &text, const std::string &option)
{
    try
    {
        std::size_t consumed = 0;
        const int value = std::stoi(text, &consumed);
        Require(consumed == text.size() && value > 0,
                option + " must be a positive integer");
        return value;
    }
    catch (const std::invalid_argument &)
    {
        throw IoError(option + " must be a positive integer", kExitInvalidInput);
    }
    catch (const std::out_of_range &)
    {
        throw IoError(option + " is out of range", kExitInvalidInput);
    }
}

nlohmann::json SelectionConfig(const YAML::Node &selection)
{
    nlohmann::json config;
    config["window_frames"] = NodeInt(selection, "window_frames", 60);
    config["step_frames"] = NodeInt(selection, "step_frames", 10);
    config["score"] = NodeString(selection, "score", "micro_dynamic_fraction_after_ego");
    config["tie_break"] = NodeString(selection, "tie_break", "earliest_window");
    return config;
}

void WriteCandidateCsv(const fs::path &path, const std::vector<WindowScore> &windows)
{
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output.good())
    {
        throw IoError("cannot write " + path.string(), kExitInvalidInput);
    }
    output << "start_frame,end_frame,frame_count,dynamic_points,static_points,score\n";
    output << std::setprecision(17);
    for (const WindowScore &window : windows)
    {
        output << window.start_frame << ',' << window.end_frame << ',' << window.frame_count << ','
               << window.dynamic_points << ',' << window.static_points << ',' << window.score
               << '\n';
    }
    if (!output.good())
    {
        throw IoError("error while writing " + path.string(), kExitInvalidInput);
    }
}

nlohmann::json PartialScores(const std::vector<FrameScore> &scores, std::string_view sequence)
{
    nlohmann::json document;
    document["schema_version"] = 1;
    document["sequence"] = std::string(sequence);
    document["generated_at_utc"] = raylidar::UtcNowIso8601();
    nlohmann::json frames = nlohmann::json::array();
    for (const FrameScore &score : scores)
    {
        nlohmann::json entry;
        entry["frame"] = score.frame;
        entry["dynamic_2d"] = score.dynamic_2d;
        entry["dynamic_bbox"] = score.dynamic_bbox;
        entry["dynamic_union"] = score.dynamic_union;
        entry["ego_roi"] = score.ego_roi;
        entry["kept_static"] = score.kept_static;
        entry["nonfinite"] = score.nonfinite;
        entry["score"] = score.score;
        entry["usable"] = score.usable;
        entry["unusable_reason"] = score.unusable_reason;
        frames.push_back(std::move(entry));
    }
    document["frames"] = std::move(frames);
    return document;
}

int Execute(std::span<const std::string_view> args)
{
    const ParsedArgs parsed =
        ParseArgs(args, {"config", "frame-index", "bboxes", "out-dir", "max-frames", "stride"},
                  {"help"});
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
    std::string bboxesXml = NodeString(config, "bboxes_xml", "");
    const std::string sequence = NodeString(config, "sequence", "");
    Require(!datasetRoot.empty(), configPath + ": dataset_root is not configured");
    Require(!lidarRoot.empty(), configPath + ": lidar_root is not configured");
    Require(!sequence.empty(), configPath + ": sequence is not configured");
    if (!parsed.Get("bboxes").empty())
    {
        bboxesXml = parsed.Get("bboxes");
    }
    Require(!bboxesXml.empty(), configPath + ": bboxes_xml is not configured");

    const std::vector<std::string> cameras =
        NodeStrings(config["cameras"], configPath + ": cameras");

    fs::path phaseDir = NodeString(config["artifacts"], "phase_dir", "");
    if (phaseDir.empty())
    {
        phaseDir = PhaseArtifactsDir(2);
    }

    const YAML::Node selection = config["selection"];
    Require(selection && selection.IsMap(), configPath + ": missing selection section");
    const int windowFrames = NodeInt(selection, "window_frames", -1);
    const int stepFrames = NodeInt(selection, "step_frames", -1);
    const int minUsableFrames = NodeInt(selection, "min_usable_frames", -1);
    const int minScenePoints = NodeInt(selection, "min_scene_points_per_frame", -1);
    Require(windowFrames > 0, configPath + ": selection.window_frames must be positive");
    Require(stepFrames > 0, configPath + ": selection.step_frames must be positive");
    Require(minUsableFrames > 0, configPath + ": selection.min_usable_frames must be positive");
    Require(minScenePoints >= 0,
            configPath + ": selection.min_scene_points_per_frame must be non-negative");

    fs::path outDir = parsed.Get("out-dir", "");
    if (outDir.empty())
    {
        outDir = phaseDir / "segment";
    }
    raylidar::EnsureDir(outDir);

    fs::path frameIndexPath = parsed.Get("frame-index", "");
    if (frameIndexPath.empty())
    {
        frameIndexPath = phaseDir / "streams" / "frame_index.json";
    }
    std::vector<int> frames = ReadUsableFrames(frameIndexPath);

    bool smoke = false;
    const std::string strideText = parsed.Get("stride", "");
    if (!strideText.empty())
    {
        const int stride = ParsePositive(strideText, "--stride");
        std::vector<int> reduced;
        for (std::size_t index = 0; index < frames.size();
             index += static_cast<std::size_t>(stride))
        {
            reduced.push_back(frames[index]);
        }
        frames = std::move(reduced);
        smoke = true;
    }
    const std::string maxFramesText = parsed.Get("max-frames", "");
    if (!maxFramesText.empty())
    {
        const int maxFrames = ParsePositive(maxFramesText, "--max-frames");
        if (frames.size() > static_cast<std::size_t>(maxFrames))
        {
            frames.resize(static_cast<std::size_t>(maxFrames));
        }
        smoke = true;
    }
    Require(!frames.empty(), "segment-select needs at least one usable frame");

    if (!smoke && frames.size() < static_cast<std::size_t>(minUsableFrames))
    {
        throw IoError("only " + std::to_string(frames.size()) + " usable frames; " +
                          std::to_string(minUsableFrames) + " are required",
                      kExitNoCandidate);
    }

    const Kitti360Calib calib = LoadKitti360Calib(fs::path(datasetRoot) / "calibration");
    const std::map<int, Eigen::Matrix4d> poses =
        LoadPoses(fs::path(datasetRoot) / "data_poses" / sequence / "poses.txt");
    const std::vector<std::int64_t> timestamps =
        LoadVelodyneTimestamps(fs::path(lidarRoot) / "timestamps.txt");
    const std::vector<DynamicBBox> boxes = LoadDynamicBBoxes(bboxesXml);

    SegmentInputs inputs;
    inputs.sequence = sequence;
    inputs.cameras = cameras;
    inputs.mask = ReadMaskOptions(config, configPath);
    inputs.windowFrames = windowFrames;
    inputs.stepFrames = stepFrames;
    inputs.minScenePointsPerFrame = minScenePoints;
    inputs.timestamps = &timestamps;

    std::vector<FrameScore> scores;
    scores.reserve(frames.size());
    const std::size_t progressEvery = smoke ? frames.size() : 250U;
    for (std::size_t index = 0; index < frames.size(); ++index)
    {
        scores.push_back(ScoreFrame(frames[index], datasetRoot, lidarRoot, calib, poses, boxes,
                                    inputs));
        const std::size_t done = index + 1U;
        if (done % progressEvery == 0U || done == frames.size())
        {
            std::cerr << "scored " << done << '/' << frames.size() << '\n' << std::flush;
        }
        if (!smoke && done % 500U == 0U)
        {
            WriteJson(outDir / "segment_scores.partial.json", PartialScores(scores, sequence));
        }
    }

    const std::vector<WindowScore> windows = CandidateWindows(scores, windowFrames, stepFrames);
    const std::optional<WindowScore> selected = SelectWindow(windows);

    nlohmann::json configDocument;
    configDocument["selection"] = SelectionConfig(selection);
    const std::string mode = smoke ? "smoke" : "full";
    const nlohmann::json manifest =
        SelectionManifest(scores, windows, selected, sequence, mode, configDocument);

    if (smoke)
    {
        WriteJson(outDir / "segment_manifest.smoke.json", manifest);
        std::cerr << "warning: smoke run wrote segment_manifest.smoke.json; this is not final\n";
        return 0;
    }

    WriteJson(outDir / "segment_manifest.json", manifest);
    WriteCandidateCsv(outDir / "candidate_windows.csv", windows);

    if (!selected.has_value())
    {
        std::cerr << "error: no complete " << windowFrames
                  << "-frame candidate window exists\n";
        return kExitNoCandidate;
    }

    std::cout << "selected_start=" << selected->start_frame
              << " selected_end=" << selected->end_frame << " score=" << std::setprecision(17)
              << selected->score << " candidates=" << windows.size() << '\n';
    return 0;
}
} // namespace

int RunSegmentSelect(std::span<const std::string_view> args)
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

RAYLIDAR_REGISTER_COMMAND("segment-select",
                          "Select the lowest-dynamic 60-frame window from the usable frames",
                          RunSegmentSelect);
