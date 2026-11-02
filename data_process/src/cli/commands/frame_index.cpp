#include <raylidar/cli/command.h>
#include <raylidar/cli/commands_phase2.h>
#include <raylidar/common/artifacts.h>
#include <raylidar/common/cli_options.h>
#include <raylidar/common/io.h>
#include <raylidar/dataset/streams.h>

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <set>
#include <span>
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
using raylidar::dataset::BuildStreamIndex;
using raylidar::dataset::StreamIndex;
using raylidar::dataset::StreamIndexToJson;
using raylidar::dataset::WriteStreamIndexCsv;

constexpr int kExitInvalidInput = 4;

void PrintHelp(std::ostream &out)
{
    out << "Usage: raylidar_cli frame-index [options]\n"
        << "\n"
        << "Intersect KITTI-360 image, Velodyne, pose, and semantics streams and\n"
        << "write a deterministic usable-frame index plus a full dropped-frame ledger.\n"
        << "\n"
        << "Options:\n"
        << "  --config <path>      Phase-2 YAML (default: configs/phase2.yaml)\n"
        << "  --root <path>        Override dataset_root from the config\n"
        << "  --lidar-root <path>  Override lidar_root from the config\n"
        << "  --sequence <name>    Override sequence from the config\n"
        << "  --out-dir <path>     Artifact directory (default: <phase_dir>/streams)\n"
        << "  --help               Show this help\n"
        << "\n"
        << "Exit codes: 0 written, 2 usage error, 4 invalid input/missing files.\n";
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

void Require(const bool condition, const std::string &message)
{
    if (!condition)
    {
        throw IoError(message, kExitInvalidInput);
    }
}

nlohmann::json DroppedFramesJson(const StreamIndex &index, std::string_view sequence)
{
    nlohmann::json document;
    document["schema_version"] = 1;
    document["generated_at_utc"] = UtcNowIso8601();
    document["sequence"] = std::string(sequence);

    nlohmann::json counts;
    counts["total"] = index.usable.size() + index.dropped.size();
    counts["usable"] = index.usable.size();
    counts["dropped"] = index.dropped.size();
    document["counts"] = std::move(counts);

    nlohmann::json dropped = nlohmann::json::array();
    for (const auto &record : index.dropped)
    {
        nlohmann::json entry;
        entry["frame"] = record.frame;
        entry["missing"] = record.missing;
        dropped.push_back(std::move(entry));
    }
    document["dropped"] = std::move(dropped);
    return document;
}

int Execute(std::span<const std::string_view> args)
{
    const ParsedArgs parsed = ParseArgs(args, {"config", "root", "lidar-root", "sequence", "out-dir"}, {"help"});
    if (parsed.Flag("help"))
    {
        PrintHelp(std::cout);
        return 0;
    }

    const std::string configPath = parsed.Get("config", "configs/phase2.yaml");
    const YAML::Node config = raylidar::ReadYaml(configPath);
    Require(config && config.IsMap(), configPath + ": expected a YAML mapping");

    const std::vector<std::string> cameras = NodeStringList(config, "cameras");
    Require(cameras == std::vector<std::string>{"image_00", "image_01"},
            configPath + ": cameras must be exactly image_00,image_01");

    const YAML::Node streams = config["streams"];
    const bool requireExactPose = NodeBool(streams, "require_exact_pose", false);
    const bool requireNextPose = NodeBool(streams, "require_next_pose", false);
    const bool requireSemantics = NodeBool(streams, "require_semantics", true);
    Require(requireExactPose, configPath + ": streams.require_exact_pose must be true");
    Require(requireNextPose, configPath + ": streams.require_next_pose must be true");

    std::string root = parsed.Get("root");
    if (root.empty())
    {
        root = NodeString(config, "dataset_root", "");
    }
    std::string lidarRoot = parsed.Get("lidar-root");
    if (lidarRoot.empty())
    {
        lidarRoot = NodeString(config, "lidar_root", "");
    }
    std::string sequence = parsed.Get("sequence");
    if (sequence.empty())
    {
        sequence = NodeString(config, "sequence", "");
    }
    Require(!root.empty(), "dataset_root is not configured; pass --root");
    Require(!lidarRoot.empty(), "lidar_root is not configured; pass --lidar-root");
    Require(!sequence.empty(), "sequence is not configured; pass --sequence");

    fs::path outDir = parsed.Get("out-dir");
    if (outDir.empty())
    {
        std::string phaseDir = NodeString(config["artifacts"], "phase_dir", "");
        outDir = phaseDir.empty() ? PhaseArtifactsDir(2) : fs::path(phaseDir);
        outDir /= "streams";
    }
    raylidar::EnsureDir(outDir);

    const StreamIndex index =
        BuildStreamIndex(root, lidarRoot, sequence, cameras, requireExactPose, requireNextPose, requireSemantics);

    WriteJson(outDir / "frame_index.json", StreamIndexToJson(index, sequence, root, lidarRoot));
    WriteStreamIndexCsv(outDir / "frame_index.csv", index);
    WriteJson(outDir / "dropped_frames.json", DroppedFramesJson(index, sequence));

    std::cout << "usable=" << index.usable.size() << " dropped=" << index.dropped.size()
              << " total=" << index.usable.size() + index.dropped.size() << '\n';
    return 0;
}
} // namespace

int RunFrameIndex(std::span<const std::string_view> args)
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

RAYLIDAR_REGISTER_COMMAND("frame-index", "Intersect KITTI-360 image, Velodyne, pose, and semantics streams",
                          RunFrameIndex);
