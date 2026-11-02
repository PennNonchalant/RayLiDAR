#include <raylidar/cli/command.h>
#include <raylidar/common/cli_options.h>
#include <raylidar/common/io.h>
#include <raylidar/dataset/preflight.h>

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <iostream>
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
using raylidar::UsageError;
using raylidar::dataset::PreflightExitCode;
using raylidar::dataset::PreflightReport;
using raylidar::dataset::PreflightReportJson;
using raylidar::dataset::RenderSummary;
using raylidar::dataset::RunPreflight;

const std::vector<std::string> kDefaultForbiddenNames = {
    "calib_cam_to_cam.txt",
    "calib_velo_to_cam.txt",
    "calib_imu_to_velo.txt",
};

void PrintHelp(std::ostream &out)
{
    out << "Usage: raylidar_cli check-dataset [options]\n"
        << "\n"
        << "Report KITTI-360 dataset readiness and reject KITTI-raw calibration.\n"
        << "\n"
        << "Options:\n"
        << "  --config <path>      Dataset YAML (default: configs/dataset.yaml)\n"
        << "  --root <path>        Override the dataset root from the config\n"
        << "  --sequence <name>    Override the sequence from the config\n"
        << "  --scan-root <path>   Extra root to scan for KITTI-raw calibration "
           "(repeatable)\n"
        << "  --json <path>        Write the JSON report to this path\n"
        << "  --strict             Escalate missing/rejected images or LiDAR to exit 2\n"
        << "  --help               Show this help\n"
        << "\n"
        << "Exit codes: 0 ready, 2 blocking missing/rejected, 3 non-blocking "
           "missing/rejected, 4 invalid input.\n";
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
} // namespace

int RunCheckDataset(std::span<const std::string_view> args)
{
    try
    {
        const ParsedArgs parsed =
            ParseArgs(args, {"config", "root", "sequence", "scan-root", "json"}, {"strict", "help"});
        if (parsed.Flag("help"))
        {
            PrintHelp(std::cout);
            return 0;
        }

        const std::string configPath = parsed.Get("config", "configs/dataset.yaml");
        const YAML::Node config = raylidar::ReadYaml(configPath);
        const YAML::Node dataset = (config && config.IsMap() && config["dataset"]) ? config["dataset"] : config;

        std::string root = parsed.Get("root");
        if (root.empty())
        {
            root = NodeString(dataset, "root", "");
        }
        std::string sequence = parsed.Get("sequence");
        if (sequence.empty())
        {
            sequence = NodeString(dataset, "sequence", "");
        }

        if (root.empty())
        {
            std::cerr << "error: dataset root is not configured; pass --root\n";
            return 4;
        }
        if (sequence.empty())
        {
            std::cerr << "error: dataset sequence is not configured; pass --sequence\n";
            return 4;
        }

        std::vector<std::string> cameras = NodeStringList(dataset, "cameras");
        if (cameras.empty())
        {
            cameras = {"image_00", "image_01"};
        }

        std::vector<std::string> forbiddenNames = NodeStringList(dataset, "forbidden_calibration_names");
        if (forbiddenNames.empty())
        {
            forbiddenNames = kDefaultForbiddenNames;
        }

        std::vector<fs::path> scanRoots;
        const std::vector<std::string> scanArguments = parsed.GetAll("scan-root");
        if (!scanArguments.empty())
        {
            for (const std::string &value : scanArguments)
            {
                scanRoots.emplace_back(value);
            }
        }
        else
        {
            for (const std::string &value : NodeStringList(dataset, "scan_roots"))
            {
                scanRoots.emplace_back(value);
            }
        }

        const bool strict = parsed.Flag("strict");
        const PreflightReport report = RunPreflight(root, sequence, cameras, forbiddenNames, scanRoots);

        std::cout << RenderSummary(report);

        const std::string jsonPath = parsed.Get("json");
        if (!jsonPath.empty())
        {
            raylidar::WriteJson(jsonPath, PreflightReportJson(report));
        }

        return PreflightExitCode(report, strict);
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
        return 4;
    }
}

RAYLIDAR_REGISTER_COMMAND("check-dataset", "Report KITTI-360 dataset readiness and reject KITTI-raw calibration",
                          RunCheckDataset);
