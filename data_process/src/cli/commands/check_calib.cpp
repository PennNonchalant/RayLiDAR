#include <raylidar/calib/frames.h>
#include <raylidar/calib/kitti360.h>
#include <raylidar/cli/command.h>
#include <raylidar/cli/commands.h>
#include <raylidar/common/artifacts.h>
#include <raylidar/common/cli_options.h>
#include <raylidar/common/io.h>

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
using raylidar::WriteJson;
using raylidar::calib::AssertRotation;
using raylidar::calib::AssertRoundtrip;
using raylidar::calib::CameraHeight;
using raylidar::calib::Compose;
using raylidar::calib::ComposeTCkRectV;
using raylidar::calib::ComposeTWCkRect;
using raylidar::calib::ComposeTWV;
using raylidar::calib::EstimateGroundPlane;
using raylidar::calib::FindCommonFrames;
using raylidar::calib::GroundPlane;
using raylidar::calib::InvertRigid;
using raylidar::calib::Kitti360Calib;
using raylidar::calib::LoadCam0ToWorld;
using raylidar::calib::LoadKitti360Calib;
using raylidar::calib::LoadPoses;
using raylidar::calib::LookupPose;
using raylidar::calib::Rect4;
using raylidar::calib::RotationOf;

constexpr int kExitAssertionFailure = 5;
constexpr int kExitInvalidInput = 4;
constexpr const char *kOk = "ok";
constexpr const char *kSkippedNoLidar = "skipped_no_lidar";
constexpr const char *kFailed = "failed";

const char *const kMatrixSuffixes[] = {"r11", "r12", "r13", "tx", "r21", "r22", "r23", "ty", "r31", "r32", "r33", "tz"};

struct Thresholds
{
    double roundtripAtol = 1e-9;
    double rotationAtol = 2e-6;
    double rotationDetAtol = 2e-6;
    double cam0WorldAtol = 1e-3;
    double heightMin = 1.4;
    double heightMax = 1.9;
    double heightStability = 0.1;
};

struct GroundOptions
{
    double radiusMin = 5.0;
    double radiusMax = 30.0;
    double zMin = -3.0;
    double zMax = 1.0;
    int azimuthBins = 240;
    double trimFraction = 0.7;
    int minInliers = 100;
};

struct CameraResult
{
    std::string camera;
    bool composed = false;
    double roundtripRaw = 0.0;
    double roundtripRect = 0.0;
    double rotationOrth = 0.0;
    double rotationDet = 0.0;
    Eigen::Matrix4d T_W_Ck_rect = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d T_Ck_rect_V = Eigen::Matrix4d::Identity();
};

struct FrameResult
{
    int frame = 0;
    int poseFrame = 0;
    int delta = 0;
    Eigen::Matrix4d T_W_I = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d T_W_V = Eigen::Matrix4d::Identity();
    std::vector<CameraResult> cameras;
    bool cam0Present = false;
    double cam0Delta = 0.0;
    bool heightPresent = false;
    double height = 0.0;
    GroundPlane plane;
    std::string heightStatus = kOk;
};

struct Report
{
    Thresholds thresholds;
    GroundOptions ground;
    std::vector<FrameResult> frames;
    std::vector<std::string> failures;
    bool roundtripOk = true;
    bool rotationOk = true;
    bool cam0Ok = true;
    bool heightOk = true;
    bool heightMissing = false;
    double maxCam0Delta = 0.0;
    double heightSpread = 0.0;
    std::string heightStatus = kOk;
    bool pass = false;
};

void PrintHelp(std::ostream &out)
{
    out << "Usage: raylidar_cli check-calib [options]\n"
        << "\n"
        << "Compose the KITTI-360 transform chain and emit calib.json,\n"
        << "frames.csv, and calib_report.json.\n"
        << "\n"
        << "Options:\n"
        << "  --config <path>       Frames/config YAML (default: configs/frames.yaml)\n"
        << "  --frames <list>       Comma-separated frames; disables --auto-frames\n"
        << "  --auto-frames <n>     Contiguous common frames to sample (default: 3)\n"
        << "  --out-dir <path>      Artifact directory (default: <artifacts>/01/calib)\n"
        << "  --root <path>         Override dataset_root from the config\n"
        << "  --sequence <name>     Override sequence from the config\n"
        << "  --help                Show this help\n"
        << "\n"
        << "Exit codes: 0 pass, 4 invalid input/config, 5 G1 assertion failure.\n";
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

std::pair<double, double> NodePair(const YAML::Node &node, const char *key, double firstFallback, double secondFallback)
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

std::vector<int> ParseFrameList(const std::string &text)
{
    std::vector<int> frames;
    std::istringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ','))
    {
        const std::string trimmed = Trim(token);
        if (trimmed.empty())
        {
            continue;
        }
        frames.push_back(ParseIntExact(trimmed, "frame"));
    }
    return frames;
}

std::string FrameName(int frame)
{
    std::ostringstream out;
    out << std::setw(10) << std::setfill('0') << frame;
    return out.str();
}

fs::path LidarPath(const fs::path &root, const std::string &sequence, int frame)
{
    return root / "data_3d_raw" / sequence / "velodyne_points" / "data" / (FrameName(frame) + ".bin");
}

bool IsRegularFile(const fs::path &path)
{
    std::error_code error;
    return fs::is_regular_file(path, error);
}

Eigen::MatrixXd ReadLidarScan(const fs::path &path)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input)
    {
        throw IoError("cannot open LiDAR scan " + path.string(), kExitInvalidInput);
    }
    const std::streamsize bytes = input.tellg();
    if (bytes <= 0 || bytes % 16 != 0)
    {
        throw IoError("LiDAR scan has an invalid size: " + path.string(), kExitInvalidInput);
    }
    input.seekg(0, std::ios::beg);
    std::vector<float> raw(static_cast<std::size_t>(bytes) / sizeof(float));
    input.read(reinterpret_cast<char *>(raw.data()), bytes);
    if (!input)
    {
        throw IoError("cannot read LiDAR scan " + path.string(), kExitInvalidInput);
    }
    const std::size_t count = raw.size() / 4U;
    Eigen::MatrixXd points(static_cast<Eigen::Index>(count), 3);
    for (std::size_t index = 0; index < count; ++index)
    {
        points(static_cast<Eigen::Index>(index), 0) = raw[index * 4U + 0U];
        points(static_cast<Eigen::Index>(index), 1) = raw[index * 4U + 1U];
        points(static_cast<Eigen::Index>(index), 2) = raw[index * 4U + 2U];
    }
    return points;
}

std::string Sci(double value)
{
    std::ostringstream out;
    out.precision(6);
    out << std::scientific << value;
    return out.str();
}

double MaxAbsDiff(const Eigen::Matrix4d &a, const Eigen::Matrix4d &b)
{
    return (a - b).cwiseAbs().maxCoeff();
}

nlohmann::json MatrixJson(const Eigen::MatrixXd &matrix)
{
    nlohmann::json rows = nlohmann::json::array();
    for (Eigen::Index row = 0; row < matrix.rows(); ++row)
    {
        nlohmann::json values = nlohmann::json::array();
        for (Eigen::Index col = 0; col < matrix.cols(); ++col)
        {
            values.push_back(matrix(row, col));
        }
        rows.push_back(values);
    }
    return rows;
}

std::string Cell(double value)
{
    std::ostringstream out;
    out << std::setprecision(17) << value;
    return out.str();
}

std::string Join(const std::vector<std::string> &cells)
{
    std::string line;
    for (std::size_t index = 0; index < cells.size(); ++index)
    {
        if (index != 0U)
        {
            line += ',';
        }
        line += cells[index];
    }
    return line;
}

void AppendMatrix12(std::vector<std::string> &cells, const Eigen::Matrix4d &matrix)
{
    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            cells.push_back(Cell(matrix(row, col)));
        }
    }
}

std::string FramesCsvHeader(const std::vector<std::string> &cameras)
{
    std::vector<std::string> cells = {"frame", "pose_frame", "pose_delta"};
    for (const char *suffix : kMatrixSuffixes)
    {
        cells.push_back(std::string("T_W_I_") + suffix);
    }
    for (const std::string &camera : cameras)
    {
        for (const char *suffix : kMatrixSuffixes)
        {
            cells.push_back(camera + "_rect_" + suffix);
        }
    }
    cells.push_back("camera_height_m");
    return Join(cells);
}

std::string FramesCsvRow(const FrameResult &frame, const std::vector<std::string> &cameras)
{
    std::vector<std::string> cells = {std::to_string(frame.frame), std::to_string(frame.poseFrame),
                                      std::to_string(frame.delta)};
    AppendMatrix12(cells, frame.T_W_I);
    for (const std::string &camera : cameras)
    {
        const auto found = std::find_if(frame.cameras.begin(), frame.cameras.end(),
                                        [&camera](const CameraResult &result) { return result.camera == camera; });
        if (found == frame.cameras.end())
        {
            for (int index = 0; index < 12; ++index)
            {
                cells.push_back("");
            }
        }
        else
        {
            AppendMatrix12(cells, found->T_W_Ck_rect);
        }
    }
    cells.push_back(frame.heightPresent ? Cell(frame.height) : "");
    return Join(cells);
}

void WriteFramesCsv(const fs::path &path, const Report &report, const std::vector<std::string> &cameras)
{
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output)
    {
        throw IoError("cannot write " + path.string(), kExitInvalidInput);
    }
    output << FramesCsvHeader(cameras) << '\n';
    for (const FrameResult &frame : report.frames)
    {
        output << FramesCsvRow(frame, cameras) << '\n';
    }
}

nlohmann::json CalibJson(const Kitti360Calib &calib, const std::vector<std::string> &cameras)
{
    nlohmann::json root;
    root["schema_version"] = 1;
    root["T_V_C0"] = MatrixJson(calib.T_V_C0);
    for (const std::string &camera : cameras)
    {
        const auto pose = calib.T_I_Ck.find(camera);
        if (pose != calib.T_I_Ck.end())
        {
            root["T_I_" + camera] = MatrixJson(pose->second);
        }
        const auto rectification = calib.R_rect.find(camera);
        if (rectification != calib.R_rect.end())
        {
            root["R_rect_" + camera] = MatrixJson(rectification->second);
        }
        const auto projection = calib.P_rect.find(camera);
        if (projection != calib.P_rect.end())
        {
            root["P_rect_" + camera] = MatrixJson(projection->second);
        }
        const auto intrinsic = calib.K_rect.find(camera);
        if (intrinsic != calib.K_rect.end())
        {
            root["K_rect_" + camera] = MatrixJson(intrinsic->second);
        }
    }
    return root;
}

nlohmann::json ReportJson(const Report &report, const std::string &sequence, const std::vector<std::string> &cameras)
{
    nlohmann::json root;
    root["schema_version"] = 1;
    root["sequence"] = sequence;
    root["cameras"] = cameras;

    nlohmann::json frames = nlohmann::json::array();
    nlohmann::json heights = nlohmann::json::array();
    for (const FrameResult &frame : report.frames)
    {
        nlohmann::json entry;
        entry["frame"] = frame.frame;
        entry["pose_frame"] = frame.poseFrame;
        entry["pose_delta"] = frame.delta;

        nlohmann::json cameraEntries = nlohmann::json::object();
        for (const CameraResult &camera : frame.cameras)
        {
            nlohmann::json values;
            values["composed"] = camera.composed;
            values["roundtrip_raw"] = camera.roundtripRaw;
            values["roundtrip_rect"] = camera.roundtripRect;
            values["rotation_orth"] = camera.rotationOrth;
            values["rotation_det"] = camera.rotationDet;
            if (camera.camera == "image_00")
            {
                values["cam0_world_delta"] = frame.cam0Delta;
            }
            cameraEntries[camera.camera] = values;
        }
        entry["cameras"] = cameraEntries;

        nlohmann::json plane;
        plane["normal"] = {frame.plane.normal.x(), frame.plane.normal.y(), frame.plane.normal.z()};
        plane["offset"] = frame.plane.offset;
        entry["ground_plane"] = plane;
        if (frame.heightPresent)
        {
            entry["camera_height_m"] = frame.height;
            heights.push_back(frame.height);
        }
        else
        {
            entry["camera_height_m"] = nullptr;
            heights.push_back(nullptr);
        }
        entry["camera_height_status"] = frame.heightStatus;
        frames.push_back(entry);
    }
    root["frames"] = frames;
    root["max_cam0_world_delta"] = report.maxCam0Delta;
    root["camera_height_m"] = heights;
    root["camera_height_status"] = report.heightStatus;
    root["camera_height_spread_m"] = report.heightSpread;

    nlohmann::json checks;
    checks["roundtrip"] = report.roundtripOk;
    checks["rotation"] = report.rotationOk;
    checks["cam0_world"] = report.cam0Ok;
    checks["camera_height"] = report.heightOk;
    root["checks"] = checks;
    root["failures"] = report.failures;
    root["pass"] = report.pass;
    return root;
}

void Fail(Report &report, bool &flag, const std::string &message)
{
    flag = false;
    report.failures.push_back(message);
}

/// Load one frame, assert its transforms, and fill `result`.
void ProcessFrame(const Kitti360Calib &calib, const std::map<int, Eigen::Matrix4d> &poses,
                  const std::map<int, Eigen::Matrix4d> &cam0ToWorld, int frame, const std::vector<std::string> &cameras,
                  int maxDelta, const Thresholds &thresholds, const GroundOptions &ground, const fs::path &root,
                  const std::string &sequence, Report &report)
{
    FrameResult result;
    result.frame = frame;

    const auto lookup = LookupPose(poses, frame, maxDelta);
    result.poseFrame = lookup.poseFrame;
    result.delta = lookup.delta;
    result.T_W_I = lookup.T_W_I;

    try
    {
        result.T_W_V = ComposeTWV(calib, lookup.T_W_I);
        AssertRotation(RotationOf(result.T_W_V), thresholds.rotationAtol, thresholds.rotationDetAtol);
    }
    catch (const std::exception &error)
    {
        Fail(report, report.rotationOk, "frame " + std::to_string(frame) + " T_W_V rotation: " + error.what());
    }

    for (const std::string &camera : cameras)
    {
        CameraResult entry;
        entry.camera = camera;

        // Stage A: compose and prove the camera-relative round trip at 1e-9.
        bool composed = false;
        try
        {
            entry.T_W_Ck_rect = ComposeTWCkRect(calib, lookup.T_W_I, camera);
            entry.T_Ck_rect_V = ComposeTCkRectV(calib, lookup.T_W_I, camera);

            const Eigen::Matrix4d T_C0_Ck = Compose(InvertRigid(calib.T_I_Ck.at("image_00")), calib.T_I_Ck.at(camera));
            const Eigen::Matrix4d T_V_Ck = Compose(calib.T_V_C0, T_C0_Ck);
            const Eigen::Matrix4d T_Ck_V = InvertRigid(T_V_Ck);
            entry.roundtripRaw = (T_Ck_V * InvertRigid(T_Ck_V) - Eigen::Matrix4d::Identity()).cwiseAbs().maxCoeff();
            AssertRoundtrip(T_Ck_V, InvertRigid(T_Ck_V), thresholds.roundtripAtol);

            const Eigen::Matrix4d fromRaw = Compose(Rect4(calib.R_rect.at(camera)), T_Ck_V);
            entry.roundtripRect = MaxAbsDiff(entry.T_Ck_rect_V, fromRaw);
            if (!(entry.roundtripRect <= thresholds.roundtripAtol))
            {
                throw std::runtime_error("rectified/raw round-trip error " + Sci(entry.roundtripRect) + " > " +
                                         Sci(thresholds.roundtripAtol));
            }
            composed = true;
        }
        catch (const IoError &)
        {
            throw;
        }
        catch (const std::out_of_range &)
        {
            throw;
        }
        catch (const std::exception &error)
        {
            Fail(report, report.roundtripOk, "frame " + std::to_string(frame) + " " + camera + ": " + error.what());
        }

        // Stage B: every composed rotation must be orthonormal with det +1.
        if (composed)
        {
            try
            {
                const Eigen::Matrix3d rotation = RotationOf(entry.T_W_Ck_rect);
                AssertRotation(rotation, thresholds.rotationAtol, thresholds.rotationDetAtol);
                const Eigen::Matrix3d composedRotation =
                    RotationOf(result.T_W_V) * RotationOf(InvertRigid(entry.T_Ck_rect_V));
                AssertRotation(composedRotation, thresholds.rotationAtol, thresholds.rotationDetAtol);
                entry.rotationOrth =
                    (rotation.transpose() * rotation - Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff();
                entry.rotationDet = std::abs(rotation.determinant() - 1.0);
                entry.composed = true;
            }
            catch (const std::exception &error)
            {
                Fail(report, report.rotationOk,
                     "frame " + std::to_string(frame) + " " + camera + " rotation: " + error.what());
            }
        }

        if (camera == "image_00")
        {
            const auto expected = cam0ToWorld.find(frame);
            if (expected == cam0ToWorld.end())
            {
                Fail(report, report.cam0Ok, "frame " + std::to_string(frame) + " is missing from cam0_to_world.txt");
            }
            else
            {
                try
                {
                    const Eigen::Matrix4d composed = ComposeTWCkRect(calib, lookup.T_W_I, "image_00");
                    result.cam0Delta = MaxAbsDiff(composed, expected->second);
                    result.cam0Present = true;
                    report.maxCam0Delta = std::max(report.maxCam0Delta, result.cam0Delta);
                    if (!(result.cam0Delta <= thresholds.cam0WorldAtol))
                    {
                        throw std::runtime_error("T_W_C0_rect vs cam0_to_world delta " + Sci(result.cam0Delta) + " > " +
                                                 Sci(thresholds.cam0WorldAtol));
                    }
                }
                catch (const std::exception &error)
                {
                    Fail(report, report.cam0Ok, "frame " + std::to_string(frame) + " cam0_to_world: " + error.what());
                }
            }
        }
        result.cameras.push_back(entry);
    }

    const fs::path scan = LidarPath(root, sequence, frame);
    if (!IsRegularFile(scan))
    {
        result.heightStatus = kSkippedNoLidar;
        report.heightMissing = true;
        Fail(report, report.heightOk,
             "frame " + std::to_string(frame) + " has no LiDAR scan: camera height cannot be measured");
    }
    else
    {
        try
        {
            const Eigen::MatrixXd pointsV = ReadLidarScan(scan);
            result.plane = EstimateGroundPlane(pointsV, ground.radiusMin, ground.radiusMax, ground.zMin, ground.zMax,
                                               ground.azimuthBins, ground.trimFraction, ground.minInliers);
            result.height = CameraHeight(calib.T_V_C0, result.plane);
            result.heightPresent = true;
            if (!(result.height >= thresholds.heightMin && result.height <= thresholds.heightMax))
            {
                throw std::runtime_error("camera height " + Sci(result.height) + " outside [" +
                                         Sci(thresholds.heightMin) + ", " + Sci(thresholds.heightMax) + "]");
            }
        }
        catch (const std::exception &error)
        {
            result.heightStatus = kFailed;
            Fail(report, report.heightOk, "frame " + std::to_string(frame) + " camera height: " + error.what());
        }
    }

    report.frames.push_back(result);
}

int Execute(std::span<const std::string_view> args)
{
    const ParsedArgs parsed =
        ParseArgs(args, {"config", "frames", "auto-frames", "out-dir", "root", "sequence"}, {"help"});
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

    std::vector<std::string> cameras = NodeStringList(config, "cameras");
    if (cameras.empty())
    {
        cameras = {"image_00", "image_01"};
    }

    Report report;
    const YAML::Node assertions = config["assertions"];
    report.thresholds.roundtripAtol = NodeDouble(assertions, "roundtrip_atol", 1e-9);
    report.thresholds.rotationAtol = NodeDouble(assertions, "rotation_atol", 2e-6);
    report.thresholds.rotationDetAtol = NodeDouble(assertions, "rotation_det_atol", 2e-6);
    report.thresholds.cam0WorldAtol = NodeDouble(assertions, "cam0_world_atol", 1e-3);
    const auto heightRange = NodePair(assertions, "camera_height_m", 1.4, 1.9);
    report.thresholds.heightMin = heightRange.first;
    report.thresholds.heightMax = heightRange.second;
    report.thresholds.heightStability = NodeDouble(assertions, "camera_height_stability_m", 0.1);

    const YAML::Node groundNode = config["ground_estimate"];
    const auto radius = NodePair(groundNode, "radius_m", 5.0, 30.0);
    const auto zRange = NodePair(groundNode, "z_range_m", -3.0, 1.0);
    report.ground.radiusMin = radius.first;
    report.ground.radiusMax = radius.second;
    report.ground.zMin = zRange.first;
    report.ground.zMax = zRange.second;
    report.ground.azimuthBins = NodeInt(groundNode, "azimuth_bins", 240);
    report.ground.trimFraction = NodeDouble(groundNode, "trim_fraction", 0.7);
    report.ground.minInliers = NodeInt(groundNode, "min_inliers", 100);

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
                throw IoError("frame " + std::to_string(frame) + " is missing from poses.txt (" +
                                  (datasetRoot / "data_poses" / sequence / "poses.txt").string() + ")",
                              kExitInvalidInput);
            }
            if (cam0ToWorld.find(frame) == cam0ToWorld.end())
            {
                throw IoError("frame " + std::to_string(frame) + " is missing from cam0_to_world.txt (" +
                                  (datasetRoot / "data_poses" / sequence / "cam0_to_world.txt").string() + ")",
                              kExitInvalidInput);
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

    for (const int frame : frames)
    {
        ProcessFrame(calib, poses, cam0ToWorld, frame, cameras, maxDelta, report.thresholds, report.ground, datasetRoot,
                     sequence, report);
    }

    // Camera-height stability across the sampled window.
    std::vector<double> heights;
    for (const FrameResult &frame : report.frames)
    {
        if (frame.heightPresent)
        {
            heights.push_back(frame.height);
        }
    }
    if (!heights.empty())
    {
        const auto [minimum, maximum] = std::minmax_element(heights.begin(), heights.end());
        report.heightSpread = *maximum - *minimum;
        if (report.heightSpread > report.thresholds.heightStability)
        {
            Fail(report, report.heightOk,
                 "camera height spread " + Sci(report.heightSpread) + " > " + Sci(report.thresholds.heightStability));
        }
    }
    else
    {
        report.heightOk = false;
    }
    if (report.heightMissing)
    {
        report.heightStatus = kSkippedNoLidar;
    }
    else if (heights.size() != report.frames.size())
    {
        report.heightStatus = kFailed;
    }
    if (report.heightStatus == kFailed || report.heightStatus == kSkippedNoLidar)
    {
        report.heightOk = false;
    }

    report.pass =
        report.failures.empty() && report.roundtripOk && report.rotationOk && report.cam0Ok && report.heightOk;

    fs::path outDir = parsed.Get("out-dir");
    if (outDir.empty())
    {
        outDir = PhaseArtifactsDir(1) / "calib";
    }
    raylidar::EnsureDir(outDir);
    WriteJson(outDir / "calib.json", CalibJson(calib, cameras));
    WriteFramesCsv(outDir / "frames.csv", report, cameras);
    WriteJson(outDir / "calib_report.json", ReportJson(report, sequence, cameras));

    if (!report.pass)
    {
        std::cerr << "check-calib: " << report.failures.size() << " failure(s)\n";
        for (const std::string &failure : report.failures)
        {
            std::cerr << "  - " << failure << '\n';
        }
        return kExitAssertionFailure;
    }

    std::cout << "check-calib: pass (" << report.frames.size() << " frame(s), max cam0 delta "
              << Sci(report.maxCam0Delta) << ", height spread " << Sci(report.heightSpread) << " m)\n";
    return 0;
}
} // namespace

int RunCheckCalib(std::span<const std::string_view> args)
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

RAYLIDAR_REGISTER_COMMAND("check-calib",
                          "Compose the KITTI-360 transform chain and emit calib.json/frames.csv/calib_report.json",
                          RunCheckCalib);
