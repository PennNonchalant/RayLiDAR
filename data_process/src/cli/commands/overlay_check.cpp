#include <raylidar/calib/frames.h>
#include <raylidar/calib/kitti360.h>
#include <raylidar/cli/command.h>
#include <raylidar/cli/commands.h>
#include <raylidar/common/artifacts.h>
#include <raylidar/common/cli_options.h>
#include <raylidar/common/io.h>
#include <raylidar/overlay/project.h>

#include <yaml-cpp/yaml.h>

#include <Eigen/Core>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
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
using raylidar::calib::ComposeTCkRectV;
using raylidar::calib::FindCommonFrames;
using raylidar::calib::Kitti360Calib;
using raylidar::calib::LoadCam0ToWorld;
using raylidar::calib::LoadKitti360Calib;
using raylidar::calib::LoadPoses;
using raylidar::calib::LookupPose;
using raylidar::overlay::OverlayFrame;

constexpr int kExitAssertionFailure = 5;
constexpr int kExitInvalidInput = 4;

void PrintHelp(std::ostream &out)
{
    out << "Usage: raylidar_cli overlay-check [options]\n"
        << "\n"
        << "Project LiDAR into rectified data_rect images and emit advisory\n"
        << "1x/4x overlay PNGs plus JSON reports. Never declares G1 passed.\n"
        << "\n"
        << "Options:\n"
        << "  --config <path>       Frames/config YAML (default: configs/frames.yaml)\n"
        << "  --frames <list>       Comma-separated frames; disables --auto-frames\n"
        << "  --auto-frames <n>     Contiguous common frames to sample (default: 3)\n"
        << "  --cams <list>         Comma-separated cameras (default: image_00,image_01)\n"
        << "  --out-dir <path>      Artifact directory (default: <artifacts>/01/overlays)\n"
        << "  --min-range <m>       Minimum camera-frame range (default: 10)\n"
        << "  --max-range <m>       Maximum camera-frame range (default: 30)\n"
        << "  --margin-px <px>      Image border margin (default: 5)\n"
        << "  --max-points <n>      Deterministic point cap (default: 300000)\n"
        << "  --root <path>         Override dataset_root from the config\n"
        << "  --sequence <name>     Override sequence from the config\n"
        << "  --help                Show this help\n"
        << "\n"
        << "Exit codes: 0 processed, 4 invalid input/missing files/non-data_rect image,\n"
        << "5 projection invariant failure.\n";
}

std::string Trim(std::string value)
{
    const auto notSpace = [](unsigned char ch) { return std::isspace(ch) == 0; };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), notSpace));
    value.erase(std::find_if(value.rbegin(), value.rend(), notSpace).base(), value.end());
    return value;
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

std::vector<std::string> NodeStringList(const YAML::Node &node, const char *key)
{
    std::vector<std::string> values;
    if (!node || !node.IsMap())
    {
        return values;
    }
    const YAML::Node sequence = node[key];
    if (!sequence || !sequence.IsSequence())
    {
        return values;
    }
    for (const YAML::Node &value : sequence)
    {
        if (value.IsScalar())
        {
            values.push_back(value.as<std::string>());
        }
    }
    return values;
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

int ParseIntExact(const std::string &text, const char *what)
{
    const std::string trimmed = Trim(text);
    if (trimmed.empty())
    {
        throw IoError(std::string("invalid ") + what + ": empty value", kExitInvalidInput);
    }
    char *end = nullptr;
    const long value = std::strtol(trimmed.c_str(), &end, 10);
    if (end == trimmed.c_str() || *end != '\0')
    {
        throw IoError(std::string("invalid ") + what + ": '" + trimmed + "'", kExitInvalidInput);
    }
    return static_cast<int>(value);
}

double ParseDoubleExact(const std::string &text, const char *what)
{
    const std::string trimmed = Trim(text);
    if (trimmed.empty())
    {
        throw IoError(std::string("invalid ") + what + ": empty value", kExitInvalidInput);
    }
    char *end = nullptr;
    const double value = std::strtod(trimmed.c_str(), &end);
    if (end == trimmed.c_str() || *end != '\0' || !std::isfinite(value))
    {
        throw IoError(std::string("invalid ") + what + ": '" + trimmed + "'", kExitInvalidInput);
    }
    return value;
}

std::vector<std::string> SplitComma(const std::string &text)
{
    std::vector<std::string> values;
    std::istringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ','))
    {
        const std::string trimmed = Trim(token);
        if (!trimmed.empty())
        {
            values.push_back(trimmed);
        }
    }
    return values;
}

std::vector<int> ParseFrameList(const std::string &text)
{
    std::vector<int> frames;
    for (const std::string &token : SplitComma(text))
    {
        const int frame = ParseIntExact(token, "frame");
        if (frame < 0)
        {
            throw IoError("invalid frame: " + token + " (must be non-negative)", kExitInvalidInput);
        }
        frames.push_back(frame);
    }
    return frames;
}

std::string FrameName(int frame)
{
    std::ostringstream out;
    out << std::setw(10) << std::setfill('0') << frame;
    return out.str();
}

bool HasComponent(const fs::path &path, std::string_view name)
{
    for (const fs::path &part : path)
    {
        if (part.string() == name)
        {
            return true;
        }
    }
    return false;
}

bool IsRegularFile(const fs::path &path)
{
    std::error_code error;
    return fs::is_regular_file(path, error);
}

fs::path ImagePath(const fs::path &root, const std::string &sequence, const std::string &camera, int frame)
{
    return root / "data_2d_raw" / sequence / camera / "data_rect" / (FrameName(frame) + ".png");
}

fs::path LidarPath(const fs::path &root, const std::string &sequence, int frame)
{
    return root / "data_3d_raw" / sequence / "velodyne_points" / "data" / (FrameName(frame) + ".bin");
}

std::vector<float> ReadLidarScan(const fs::path &path)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input)
    {
        throw IoError("cannot open LiDAR scan " + path.string(), kExitInvalidInput);
    }
    const std::streamsize bytes = input.tellg();
    if (bytes <= 0 || bytes % 16 != 0)
    {
        throw IoError("LiDAR scan has an invalid size (expected a positive multiple of 16 bytes): " + path.string(),
                      kExitInvalidInput);
    }
    input.seekg(0, std::ios::beg);
    std::vector<float> raw(static_cast<std::size_t>(bytes) / sizeof(float));
    input.read(reinterpret_cast<char *>(raw.data()), bytes);
    if (!input)
    {
        throw IoError("cannot read LiDAR scan " + path.string(), kExitInvalidInput);
    }
    return raw;
}

Eigen::MatrixXd BuildPoints(const std::vector<float> &raw, int maxPoints, bool &subsampled)
{
    const std::size_t count = raw.size() / 4U;
    subsampled = false;
    std::size_t target = count;
    if (maxPoints > 0 && count > static_cast<std::size_t>(maxPoints))
    {
        target = static_cast<std::size_t>(maxPoints);
        subsampled = true;
    }
    Eigen::MatrixXd points(static_cast<Eigen::Index>(target), 3);
    if (!subsampled)
    {
        for (std::size_t index = 0; index < count; ++index)
        {
            points(static_cast<Eigen::Index>(index), 0) = raw[index * 4U + 0U];
            points(static_cast<Eigen::Index>(index), 1) = raw[index * 4U + 1U];
            points(static_cast<Eigen::Index>(index), 2) = raw[index * 4U + 2U];
        }
        return points;
    }
    for (std::size_t index = 0; index < target; ++index)
    {
        const std::size_t source =
            target == 1U
                ? 0U
                : static_cast<std::size_t>(std::llround(static_cast<double>(index) * static_cast<double>(count - 1U) /
                                                        static_cast<double>(target - 1U)));
        points(static_cast<Eigen::Index>(index), 0) = raw[source * 4U + 0U];
        points(static_cast<Eigen::Index>(index), 1) = raw[source * 4U + 1U];
        points(static_cast<Eigen::Index>(index), 2) = raw[source * 4U + 2U];
    }
    return points;
}

std::string Cell(double value)
{
    std::ostringstream out;
    out.precision(3);
    out << std::fixed << value;
    return out.str();
}

int Execute(std::span<const std::string_view> args)
{
    const ParsedArgs parsed = ParseArgs(args,
                                        {"config", "frames", "auto-frames", "cams", "out-dir", "min-range", "max-range",
                                         "margin-px", "max-points", "root", "sequence"},
                                        {"help"});
    if (parsed.Flag("help"))
    {
        PrintHelp(std::cout);
        return 0;
    }

    const std::string configPath = parsed.Get("config", "configs/frames.yaml");
    const YAML::Node config = raylidar::ReadYaml(configPath);
    if (!config || !config.IsMap())
    {
        throw IoError(configPath + ": expected a YAML mapping", kExitInvalidInput);
    }

    std::string root = parsed.Get("root");
    if (root.empty())
    {
        root = NodeString(config, "dataset_root", "");
    }
    std::string sequence = parsed.Get("sequence");
    if (sequence.empty())
    {
        sequence = NodeString(config, "sequence", "");
    }
    if (root.empty())
    {
        throw IoError("dataset_root is not configured; pass --root", kExitInvalidInput);
    }
    if (sequence.empty())
    {
        throw IoError("sequence is not configured; pass --sequence", kExitInvalidInput);
    }

    std::vector<std::string> panel = NodeStringList(config, "cameras");
    if (panel.empty())
    {
        panel = {"image_00", "image_01"};
    }
    std::vector<std::string> cameras = SplitComma(parsed.Get("cams", "image_00,image_01"));
    if (cameras.empty())
    {
        throw IoError("--cams did not contain any camera", kExitInvalidInput);
    }
    for (const std::string &camera : cameras)
    {
        if (std::find(panel.begin(), panel.end(), camera) == panel.end())
        {
            throw IoError(
                "unsupported camera '" + camera + "' (allowed: " +
                    [&panel]() {
                        std::string list;
                        for (std::size_t index = 0; index < panel.size(); ++index)
                        {
                            if (index != 0U)
                            {
                                list += ", ";
                            }
                            list += panel[index];
                        }
                        return list;
                    }() +
                    ")",
                kExitInvalidInput);
        }
    }

    const YAML::Node overlayConfig = config["overlay"];
    const double configuredMin = NodeDouble(overlayConfig, "min_range_m", 10.0);
    const double configuredMax = NodeDouble(overlayConfig, "max_range_m", 30.0);
    const int configuredMargin = NodeInt(overlayConfig, "image_margin_px", 5);

    double minRange = configuredMin;
    double maxRange = configuredMax;
    int marginPx = configuredMargin;
    if (parsed.Has("min-range"))
    {
        minRange = ParseDoubleExact(parsed.Get("min-range"), "min-range");
    }
    if (parsed.Has("max-range"))
    {
        maxRange = ParseDoubleExact(parsed.Get("max-range"), "max-range");
    }
    if (parsed.Has("margin-px"))
    {
        marginPx = ParseIntExact(parsed.Get("margin-px"), "margin-px");
    }
    if (!(maxRange > minRange))
    {
        throw IoError("--max-range must be greater than --min-range", kExitInvalidInput);
    }
    if (marginPx < 0)
    {
        throw IoError("--margin-px must be non-negative", kExitInvalidInput);
    }
    int maxPoints = 300000;
    if (parsed.Has("max-points"))
    {
        maxPoints = ParseIntExact(parsed.Get("max-points"), "max-points");
    }
    if (maxPoints <= 0)
    {
        throw IoError("--max-points must be positive", kExitInvalidInput);
    }

    const int maxDelta = NodeInt(config["pose_lookup"], "max_delta_frames", 5);
    const fs::path datasetRoot(root);
    const auto calib = LoadKitti360Calib(datasetRoot / "calibration");
    const auto poses = LoadPoses(datasetRoot / "data_poses" / sequence / "poses.txt");
    const auto cam0ToWorld = LoadCam0ToWorld(datasetRoot / "data_poses" / sequence / "cam0_to_world.txt");

    std::vector<int> frames;
    if (parsed.Has("frames"))
    {
        frames = ParseFrameList(parsed.Get("frames"));
        if (frames.empty())
        {
            throw IoError("--frames did not contain any frame ids", kExitInvalidInput);
        }
        for (const int frame : frames)
        {
            if (poses.find(frame) == poses.end())
            {
                throw IoError("frame " + std::to_string(frame) + " is missing from poses.txt", kExitInvalidInput);
            }
            if (cam0ToWorld.find(frame) == cam0ToWorld.end())
            {
                throw IoError("frame " + std::to_string(frame) + " is missing from cam0_to_world.txt",
                              kExitInvalidInput);
            }
            for (const std::string &camera : cameras)
            {
                if (!IsRegularFile(ImagePath(datasetRoot, sequence, camera, frame)))
                {
                    throw IoError("frame " + std::to_string(frame) + " " + camera + " image is missing",
                                  kExitInvalidInput);
                }
            }
            if (!IsRegularFile(LidarPath(datasetRoot, sequence, frame)))
            {
                throw IoError("frame " + std::to_string(frame) + " LiDAR scan is missing", kExitInvalidInput);
            }
        }
    }
    else
    {
        int requested = NodeInt(config["frame_sampling"], "count", 3);
        if (parsed.Has("auto-frames"))
        {
            requested = ParseIntExact(parsed.Get("auto-frames"), "auto-frames");
        }
        if (requested <= 0)
        {
            throw IoError("--auto-frames must be positive", kExitInvalidInput);
        }
        frames =
            FindCommonFrames(datasetRoot, sequence, poses, cam0ToWorld, cameras, static_cast<std::size_t>(requested));
        if (frames.empty())
        {
            throw IoError("no common frames are available for sequence " + sequence, kExitInvalidInput);
        }
        if (frames.size() < static_cast<std::size_t>(requested))
        {
            std::cerr << "warning: only " << frames.size() << " common frame(s) available (" << requested
                      << " requested)\n";
        }
    }

    fs::path outDir = parsed.Get("out-dir");
    if (outDir.empty())
    {
        outDir = PhaseArtifactsDir(1) / "overlays";
    }
    raylidar::EnsureDir(outDir);

    std::cout << std::left << std::setw(12) << "frame" << std::setw(12) << "camera" << std::setw(10) << "valid"
              << std::setw(12) << "median_px" << std::setw(12) << "p90_px" << std::setw(14) << "frac<2px" << '\n';

    nlohmann::json results = nlohmann::json::array();
    bool invariantOk = true;
    for (const int frame : frames)
    {
        const auto lookup = LookupPose(poses, frame, maxDelta);
        for (const std::string &camera : cameras)
        {
            const fs::path image = ImagePath(datasetRoot, sequence, camera, frame);
            if (!HasComponent(image, "data_rect"))
            {
                throw IoError("rectified_data_rect_only violation: logical path is not inside "
                              "data_rect: " +
                                  image.string(),
                              kExitInvalidInput);
            }
            if (!IsRegularFile(image))
            {
                throw IoError("image is missing: " + image.string(), kExitInvalidInput);
            }
            std::error_code error;
            const fs::path resolved = fs::weakly_canonical(image, error);
            if (!error && !HasComponent(resolved, "data_rect"))
            {
                throw IoError("rectified_data_rect_only violation: resolved path is not inside "
                              "data_rect: " +
                                  resolved.string(),
                              kExitInvalidInput);
            }

            const fs::path lidar = LidarPath(datasetRoot, sequence, frame);
            if (!IsRegularFile(lidar))
            {
                throw IoError("LiDAR scan is missing: " + lidar.string(), kExitInvalidInput);
            }
            const std::vector<float> raw = ReadLidarScan(lidar);
            bool subsampled = false;
            const Eigen::MatrixXd points = BuildPoints(raw, maxPoints, subsampled);

            const Eigen::Matrix4d transform = ComposeTCkRectV(calib, lookup.T_W_I, camera);
            const auto intrinsic = calib.K_rect.find(camera);
            if (intrinsic == calib.K_rect.end())
            {
                throw IoError("calibration is missing K_rect for " + camera, kExitInvalidInput);
            }

            const nlohmann::json report = OverlayFrame(image, points, transform, intrinsic->second, outDir, frame,
                                                       camera, 4, minRange, maxRange, marginPx);

            nlohmann::json entry = report;
            entry["frame"] = frame;
            entry["camera"] = camera;
            entry["subsampled"] = subsampled;

            WriteJson(outDir / ("frame_" + FrameName(frame)) / (camera + "_overlay_report.json"), entry);

            const double median = report["median_px"].is_number() ? report["median_px"].get<double>()
                                                                  : std::numeric_limits<double>::quiet_NaN();
            const double p90 = report["p90_px"].is_number() ? report["p90_px"].get<double>()
                                                            : std::numeric_limits<double>::quiet_NaN();
            const double fraction = report.value("fraction_within_2px", 0.0);
            std::cout << std::left << std::setw(12) << frame << std::setw(12) << camera << std::setw(10)
                      << report.value("points_valid", 0U) << std::setw(12) << Cell(median) << std::setw(12) << Cell(p90)
                      << std::setw(14) << Cell(fraction) << '\n';

            if (!report.value("projection_invariant", true))
            {
                invariantOk = false;
            }
            results.push_back(entry);
        }
    }

    nlohmann::json aggregate;
    aggregate["schema_version"] = 1;
    aggregate["sequence"] = sequence;
    aggregate["frames"] = frames;
    aggregate["cams"] = cameras;
    aggregate["advisory_only"] = true;
    aggregate["rectified_data_rect_only"] = true;
    aggregate["results"] = results;
    aggregate["generated_at_utc"] = UtcNowIso8601();
    WriteJson(outDir / "overlay_report.json", aggregate);

    std::cout << "ADVISORY ONLY — visual sign-off required\n";
    if (!invariantOk)
    {
        std::cerr << "overlay-check: a point marked valid fell outside the image after rounding\n";
        return kExitAssertionFailure;
    }
    return 0;
}
} // namespace

int RunOverlayCheck(std::span<const std::string_view> args)
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

RAYLIDAR_REGISTER_COMMAND("overlay-check", "Project LiDAR into rectified images and emit advisory overlay evidence",
                          RunOverlayCheck);
